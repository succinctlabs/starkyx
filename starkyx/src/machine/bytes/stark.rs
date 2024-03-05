use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::RichField;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::challenger::{Challenger, RecursiveChallenger};
use plonky2::iop::target::Target;
use plonky2::iop::witness::WitnessWrite;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use serde::{Deserialize, Serialize};

use super::air::{get_preprocessed_byte_trace, ByteAir, ByteParameters};
use super::proof::{
    ByteStarkChallenges, ByteStarkChallengesTarget, ByteStarkProof, ByteStarkProofTarget,
};
use crate::chip::trace::data::AirTraceData;
use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
use crate::chip::uint::bytes::lookup_table::multiplicity_data::ByteMultiplicityData;
use crate::chip::uint::bytes::lookup_table::table::ByteLogLookupTable;
use crate::chip::uint::bytes::operations::NUM_BIT_OPPS;
use crate::chip::{AirParameters, Chip};
use crate::machine::bytes::builder::NUM_LOOKUP_ROWS;
use crate::maybe_rayon::*;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::prover::{AirCommitment, StarkyProver};
use crate::plonky2::stark::verifier::{
    add_virtual_air_proof, set_air_proof_target, StarkyVerifier,
};
use crate::plonky2::stark::Starky;
use crate::plonky2::Plonky2Air;
use crate::trace::AirTrace;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ByteStark<L: AirParameters, C, const D: usize>
where
    L::Field: RichField,
    C: CurtaConfig<D, F = L::Field>,
{
    pub config: StarkyConfig<C, D>,
    pub stark: Starky<Chip<L>>,
    pub air_data: AirTraceData<L>,
    pub byte_trace_cap: MerkleCap<L::Field, C::Hasher>,
    pub(crate) multiplicity_data: ByteMultiplicityData,
    pub(crate) lookup_config: StarkyConfig<C, D>,
    pub(crate) lookup_stark: Starky<ByteAir<L::Field, L::CubicParams>>,
    pub(crate) lookup_air_data: AirTraceData<ByteParameters<L::Field, L::CubicParams>>,
    pub(crate) lookup_table: ByteLogLookupTable<L::Field, L::CubicParams>,
}

impl<L: AirParameters, C, const D: usize> ByteStark<L, C, D>
where
    L::Field: RichField + Extendable<D>,
    C: CurtaConfig<D, F = L::Field, FE = <L::Field as Extendable<D>>::Extension>,
    Chip<L>: Plonky2Air<L::Field, D>,
{
    pub const fn stark(&self) -> &Starky<Chip<L>> {
        &self.stark
    }

    pub const fn config(&self) -> &StarkyConfig<C, D> {
        &self.config
    }

    pub const fn lookup_stark(&self) -> &Starky<ByteAir<L::Field, L::CubicParams>> {
        &self.lookup_stark
    }

    pub const fn lookup_config(&self) -> &StarkyConfig<C, D> {
        &self.lookup_config
    }

    fn get_preprocessed_byte_trace(
        &self,
        lookup_writer: &TraceWriter<L::Field>,
    ) -> PolynomialBatch<L::Field, C::GenericConfig, D> {
        get_preprocessed_byte_trace(lookup_writer, &self.lookup_config, &self.lookup_stark)
    }

    fn generate_execution_traces(
        &self,
        execution_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
    ) -> (TraceWriter<L::Field>, TraceWriter<L::Field>) {
        // Initialize writers.
        let main_writer = TraceWriter::new(&self.air_data, execution_trace.height());
        let lookup_writer = TraceWriter::new(&self.lookup_air_data, NUM_LOOKUP_ROWS);

        // Insert execution trace and into main writer.
        let execution_trace_length = self.stark.air.execution_trace_length;
        main_writer
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .zip(execution_trace.rows_par())
            .for_each(|(row, execution_row)| {
                row[0..execution_trace_length]
                    .copy_from_slice(&execution_row[0..execution_trace_length]);
            });
        // Insert public inputs into both writers.
        main_writer
            .public_mut()
            .unwrap()
            .copy_from_slice(public_values);
        lookup_writer
            .public_mut()
            .unwrap()
            .copy_from_slice(public_values);

        // Write lookup table values
        self.lookup_table.write_table_entries(&lookup_writer);
        for i in 0..NUM_LOOKUP_ROWS {
            lookup_writer.write_row_instructions(&self.lookup_air_data, i);
        }
        // Write multiplicities
        let multiplicities = self.multiplicity_data.get_multiplicities(&main_writer);
        lookup_writer
            .write_lookup_multiplicities(self.lookup_table.multiplicities(), &[multiplicities]);

        (main_writer, lookup_writer)
    }

    fn generate_extended_traces(
        &self,
        main_writer: &TraceWriter<L::Field>,
        lookup_writer: &TraceWriter<L::Field>,
    ) {
        // Write the extended trace values
        self.lookup_air_data.write_extended_trace(lookup_writer);

        // Update global values
        main_writer
            .global
            .write()
            .unwrap()
            .copy_from_slice(&lookup_writer.global.read().unwrap());

        self.air_data.write_extended_trace(main_writer);

        // Update global values
        lookup_writer
            .global
            .write()
            .unwrap()
            .copy_from_slice(&main_writer.global.read().unwrap());
    }

    fn generate_trace(
        &self,
        execution_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
        challenger: &mut Challenger<L::Field, C::Hasher>,
        timing: &mut TimingTree,
    ) -> (AirCommitment<L::Field, C, D>, AirCommitment<L::Field, C, D>) {
        // Absorve public values into the challenger.
        challenger.observe_elements(public_values);

        // Generate execution traces.
        let (main_writer, lookup_writer) =
            self.generate_execution_traces(execution_trace, public_values);

        let main_execution_trace_values = main_writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[0..self.stark.air.execution_trace_length].to_vec())
            .collect::<Vec<_>>();
        let main_execution_trace = AirTrace {
            values: main_execution_trace_values,
            width: self.stark.air.execution_trace_length,
        };

        let lookup_preprocessed_commitment = timed!(
            timing,
            "Preprocess lookup trace",
            self.get_preprocessed_byte_trace(&lookup_writer)
        );

        let lookup_multiplicity_trace_values = lookup_writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[0..NUM_BIT_OPPS + 1].to_vec())
            .collect::<Vec<_>>();

        let lookup_multiplicity_trace = AirTrace {
            values: lookup_multiplicity_trace_values,
            width: NUM_BIT_OPPS + 1,
        };

        // Commit to execution traces
        let main_execution_commitment = timed!(
            timing,
            "Commit to execution trace",
            self.config.commit(&main_execution_trace, timing)
        );

        let lookup_multiplicity_commitment = timed!(
            timing,
            "Commit to lookup execution trace",
            self.lookup_config
                .commit(&lookup_multiplicity_trace, timing)
        );

        // Absorve the trace commitments into the challenger.
        challenger.observe_cap(&lookup_preprocessed_commitment.merkle_tree.cap);
        challenger.observe_cap(&main_execution_commitment.merkle_tree.cap);
        challenger.observe_cap(&lookup_multiplicity_commitment.merkle_tree.cap);

        // Get random AIR challenges.
        let challenges = challenger.get_n_challenges(self.stark.air.num_challenges);
        // Save challenges to both writers.
        main_writer
            .challenges
            .write()
            .unwrap()
            .extend_from_slice(&challenges);
        lookup_writer
            .challenges
            .write()
            .unwrap()
            .extend_from_slice(&challenges);

        // Generate extended traces.
        self.generate_extended_traces(&main_writer, &lookup_writer);

        let InnerWriterData {
            trace: main_trace,
            public: main_public,
            global: main_global,
            challenges: main_challenges,
            ..
        } = main_writer.into_inner().unwrap();
        let InnerWriterData {
            trace: lookup_trace,
            public: lookup_public,
            global: lookup_global,
            challenges: global_challenges,
            ..
        } = lookup_writer.into_inner().unwrap();

        // Commit to extended traces.
        let main_extended_trace_values = main_trace
            .rows_par()
            .flat_map(|row| row[self.stark.air.execution_trace_length..].to_vec())
            .collect::<Vec<_>>();
        let main_extended_trace = AirTrace {
            values: main_extended_trace_values,
            width: L::num_columns() - self.stark.air.execution_trace_length,
        };
        let main_extended_commitment = timed!(
            timing,
            "Commit to extended trace",
            self.config.commit(&main_extended_trace, timing)
        );

        let lookup_extended_trace_values = lookup_trace
            .rows_par()
            .flat_map(|row| row[self.lookup_stark.air.0.execution_trace_length..].to_vec())
            .collect::<Vec<_>>();
        let lookup_extended_trace = AirTrace {
            values: lookup_extended_trace_values,
            width: ByteParameters::<L::Field, L::CubicParams>::num_columns()
                - self.lookup_stark.air.0.execution_trace_length,
        };
        let lookup_extended_commitment = timed!(
            timing,
            "Commit to lookup extended trace",
            self.lookup_config.commit(&lookup_extended_trace, timing)
        );

        // Obsderve global values.
        challenger.observe_elements(&main_global);
        // Observe extended trace commitments.
        challenger.observe_cap(&main_extended_commitment.merkle_tree.cap);
        challenger.observe_cap(&lookup_extended_commitment.merkle_tree.cap);

        // Return the air commitments.
        (
            AirCommitment {
                trace_commitments: vec![main_execution_commitment, main_extended_commitment],
                public_inputs: main_public,
                global_values: main_global,
                challenges: main_challenges,
            },
            AirCommitment {
                trace_commitments: vec![
                    lookup_multiplicity_commitment,
                    lookup_preprocessed_commitment,
                    lookup_extended_commitment,
                ],
                public_inputs: lookup_public,
                global_values: lookup_global,
                challenges: global_challenges,
            },
        )
    }

    pub fn prove(
        &self,
        execution_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
        timing: &mut TimingTree,
    ) -> Result<ByteStarkProof<L::Field, C, D>> {
        // Initialize challenger.
        let mut challenger = Challenger::new();

        // Generate stark commitment.
        let (main_air_commitment, lookup_air_commitment) = timed!(
            timing,
            "Generate stark trace",
            self.generate_trace(execution_trace, public_values, &mut challenger, timing)
        );

        // Generate individual stark proofs.
        let main_proof = timed!(
            timing,
            "Generate main proof",
            StarkyProver::prove_with_trace(
                &self.config,
                &self.stark,
                main_air_commitment,
                &mut challenger,
                &mut TimingTree::default(),
            )?
        );

        let lookup_proof = timed!(
            timing,
            "Generate lookup proof",
            StarkyProver::prove_with_trace(
                &self.lookup_config,
                &self.lookup_stark,
                lookup_air_commitment,
                &mut challenger,
                &mut TimingTree::default(),
            )?
        );

        // Return the proof.
        Ok(ByteStarkProof {
            main_proof: main_proof.air_proof,
            lookup_proof: lookup_proof.air_proof,
            global_values: lookup_proof.global_values,
        })
    }

    pub fn get_challenges(
        &self,
        proof: &ByteStarkProof<L::Field, C, D>,
        public_values: &[L::Field],
    ) -> ByteStarkChallenges<L::Field, D> {
        // Initialize challenger.
        let mut challenger = Challenger::<L::Field, C::Hasher>::new();

        // Observe public values.
        challenger.observe_elements(public_values);

        // Observe preprocessesd trace commitment.
        challenger.observe_cap(&proof.lookup_proof.trace_caps[1]);

        // Observe execution trace commitments.
        challenger.observe_cap(&proof.main_proof.trace_caps[0]);
        challenger.observe_cap(&proof.lookup_proof.trace_caps[0]);

        // Get challenges.
        let challenges = challenger.get_n_challenges(self.stark.air.num_challenges);

        // Observe global values.
        challenger.observe_elements(&proof.global_values);
        // Observe extended trace commitments.
        challenger.observe_cap(&proof.main_proof.trace_caps[1]);
        challenger.observe_cap(&proof.lookup_proof.trace_caps[2]);

        // Get all challenges.
        let main_challenges = proof.main_proof.get_iop_challenges(
            &self.config,
            self.config.degree_bits,
            challenges.clone(),
            &mut challenger,
        );
        let lookup_challenges = proof.lookup_proof.get_iop_challenges(
            &self.lookup_config,
            self.lookup_config.degree_bits,
            challenges,
            &mut challenger,
        );

        ByteStarkChallenges {
            main_challenges,
            lookup_challenges,
        }
    }

    pub fn verify(
        &self,
        proof: ByteStarkProof<L::Field, C, D>,
        public_values: &[L::Field],
    ) -> Result<()> {
        let ByteStarkChallenges {
            main_challenges,
            lookup_challenges,
        } = self.get_challenges(&proof, public_values);

        let ByteStarkProof {
            main_proof,
            lookup_proof,
            global_values,
        } = proof;

        // Verify that the byte lookup table matches the preprocessed value.
        assert_eq!(lookup_proof.trace_caps[1], self.byte_trace_cap);

        // Verify the main AIR proof.
        StarkyVerifier::verify_with_challenges(
            &self.config,
            &self.stark,
            main_proof,
            public_values,
            &global_values,
            main_challenges,
        )?;
        // Verify the lookup AIR proof.
        StarkyVerifier::verify_with_challenges(
            &self.lookup_config,
            &self.lookup_stark,
            lookup_proof,
            public_values,
            &global_values,
            lookup_challenges,
        )
    }

    pub fn add_virtual_proof_with_pis_target(
        &self,
        builder: &mut CircuitBuilder<L::Field, D>,
    ) -> (ByteStarkProofTarget<D>, Vec<Target>) {
        let main_proof = add_virtual_air_proof(builder, &self.stark, &self.config);
        let lookup_proof = add_virtual_air_proof(builder, &self.lookup_stark, &self.lookup_config);

        let num_global_values = self.stark.air.num_global_values;
        let global_values = builder.add_virtual_targets(num_global_values);
        let public_inputs = builder.add_virtual_targets(self.stark.air.num_public_values);

        (
            ByteStarkProofTarget {
                main_proof,
                lookup_proof,
                global_values,
            },
            public_inputs,
        )
    }

    pub fn get_challenges_target(
        &self,
        builder: &mut CircuitBuilder<L::Field, D>,
        proof: &ByteStarkProofTarget<D>,
        public_values: &[Target],
    ) -> ByteStarkChallengesTarget<D> {
        // Initialize challenger.
        let mut challenger = RecursiveChallenger::<L::Field, C::InnerHasher, D>::new(builder);

        // Observe public values.
        challenger.observe_elements(public_values);

        // Observe preprocessesd trace commitment.
        challenger.observe_cap(&proof.lookup_proof.trace_caps[1]);

        // Observe execution trace commitments.
        challenger.observe_cap(&proof.main_proof.trace_caps[0]);
        challenger.observe_cap(&proof.lookup_proof.trace_caps[0]);

        // Get challenges.
        let challenges = challenger.get_n_challenges(builder, self.stark.air.num_challenges);

        // Observe global values.
        challenger.observe_elements(&proof.global_values);
        // Observe extended trace commitments.
        challenger.observe_cap(&proof.main_proof.trace_caps[1]);
        challenger.observe_cap(&proof.lookup_proof.trace_caps[2]);

        // Get all challenges.
        let main_challenges = proof.main_proof.get_iop_challenges_target(
            builder,
            &self.config,
            challenges.clone(),
            &mut challenger,
        );
        let lookup_challenges = proof.lookup_proof.get_iop_challenges_target(
            builder,
            &self.lookup_config,
            challenges,
            &mut challenger,
        );

        ByteStarkChallengesTarget {
            main_challenges,
            lookup_challenges,
        }
    }

    pub fn verify_circuit(
        &self,
        builder: &mut CircuitBuilder<L::Field, D>,
        proof: &ByteStarkProofTarget<D>,
        public_values: &[Target],
    ) {
        let challenges = self.get_challenges_target(builder, proof, public_values);
        let ByteStarkProofTarget {
            main_proof,
            lookup_proof,
            global_values,
        } = proof;

        // Verify that the byte lookup table matches the preprocessed value.
        let expected_cap = builder.constant_merkle_cap(&self.byte_trace_cap);
        builder.connect_merkle_caps(&expected_cap, &lookup_proof.trace_caps[1]);

        StarkyVerifier::verify_with_challenges_circuit(
            builder,
            &self.config,
            &self.stark,
            main_proof,
            public_values,
            global_values,
            challenges.main_challenges,
        );

        StarkyVerifier::verify_with_challenges_circuit(
            builder,
            &self.lookup_config,
            &self.lookup_stark,
            lookup_proof,
            public_values,
            global_values,
            challenges.lookup_challenges,
        )
    }

    pub fn set_proof_target<W: WitnessWrite<L::Field>>(
        &self,
        witness: &mut W,
        proof_tagret: &ByteStarkProofTarget<D>,
        proof: ByteStarkProof<L::Field, C, D>,
    ) {
        let ByteStarkProofTarget {
            main_proof,
            lookup_proof,
            global_values,
        } = proof_tagret;

        set_air_proof_target(witness, main_proof, &proof.main_proof);
        set_air_proof_target(witness, lookup_proof, &proof.lookup_proof);

        witness.set_target_arr(global_values, &proof.global_values);
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use rand::Rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::memory::time::Time;
    use crate::chip::register::element::ElementRegister;
    use crate::chip::register::Register;
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::uint::register::U32Register;
    use crate::chip::uint::util::u32_to_le_field_bytes;
    use crate::machine::builder::Builder;
    use crate::machine::bytes::builder::BytesBuilder;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::math::prelude::*;
    use crate::plonky2::stark::config::CurtaPoseidonGoldilocksConfig;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ByteTest;

    impl AirParameters for ByteTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 13;
        const EXTENDED_COLUMNS: usize = 24;
    }

    #[test]
    fn test_byte_multi_stark() {
        type L = ByteTest;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_byte_multi_stark", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let a = builder.alloc::<U32Register>();
        let b = builder.alloc::<U32Register>();

        let num_ops = 1;
        for _ in 0..num_ops {
            let _ = builder.and(&a, &b);
        }

        let num_rows = 1 << 5;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let mut rng = rand::thread_rng();
        for i in 0..num_rows {
            let a_val = rng.gen::<u32>();
            let b_val = rng.gen::<u32>();
            writer.write(&a, &u32_to_le_field_bytes(a_val), i);
            writer.write(&b, &u32_to_le_field_bytes(b_val), i);
            writer.write_row_instructions(&stark.air_data, i);
        }

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);

        let data = recursive_builder.build::<Config>();

        let mut pw = PartialWitness::new();

        pw.set_target_arr(&public_input, &public);
        stark.set_proof_target(&mut pw, &proof_target, proof);

        let rec_proof = data.prove(pw).unwrap();
        data.verify(rec_proof).unwrap();

        timing.print();
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ByteMemTest;

    impl AirParameters for ByteMemTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 13;
        const EXTENDED_COLUMNS: usize = 33;
    }

    #[test]
    fn test_byte_memory_multi_stark() {
        type L = ByteMemTest;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_byte_multi_stark", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let a_initial_value = builder.api.alloc_public::<U32Register>();

        let a_ptr = builder
            .api
            .initialize::<U32Register>(&a_initial_value, &Time::zero(), None);

        let clk = Time::from_element(builder.clk);

        let a = builder.load(&a_ptr, &clk, None, None);
        let b = builder.alloc::<U32Register>();
        let c = builder.and(&a, &b);
        builder.store(&a_ptr, c, &clk.advance(), None, None, None);

        let a_final = builder.alloc_public::<U32Register>();

        let num_rows = 1 << 5;

        builder.free(&a_ptr, a_final, &Time::constant(num_rows));
        builder.set_to_expression_last_row(&a_final, c.expr());

        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let mut rng = rand::thread_rng();

        let a_val = u32_to_le_field_bytes(rng.gen::<u32>());
        writer.write(&a_initial_value, &a_val, 0);
        writer.write_global_instructions(&stark.air_data);
        for i in 0..num_rows {
            let b_val = rng.gen::<u32>();
            writer.write(&b, &u32_to_le_field_bytes(b_val), i);
            writer.write_row_instructions(&stark.air_data, i);
        }

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);

        let data = recursive_builder.build::<Config>();

        let mut pw = PartialWitness::new();

        pw.set_target_arr(&public_input, &public);
        stark.set_proof_target(&mut pw, &proof_target, proof);

        let rec_proof = data.prove(pw).unwrap();
        data.verify(rec_proof).unwrap();

        timing.print();
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ByteSliceMemTest;

    impl AirParameters for ByteSliceMemTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 14;
        const EXTENDED_COLUMNS: usize = 36;
    }

    #[test]
    fn test_byte_slice_memory_multi_stark() {
        type L = ByteSliceMemTest;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_byte_multi_stark", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let a_init = builder.alloc_array_public::<U32Register>(4);

        let num_rows = 1 << 20;

        let a_ptr = builder.initialize_slice::<U32Register>(&a_init, &Time::zero(), None);

        let num_rows_reg =
            builder.constant::<ElementRegister>(&GoldilocksField::from_canonical_usize(num_rows));

        builder.store(
            &a_ptr.get(1),
            a_init.get(1),
            &Time::zero(),
            Some(num_rows_reg),
            None,
            None,
        );

        let clk = Time::from_element(builder.clk);
        let zero = builder.constant::<ElementRegister>(&GoldilocksField::ZERO);

        let a_0 = a_ptr.get_at(zero);
        let zero_trace = builder.alloc::<ElementRegister>();
        builder.set_to_expression(&zero_trace, GoldilocksField::ZERO.into());
        let a_0_trace = a_ptr.get_at(zero_trace);
        let a = builder.load(&a_0_trace, &clk, None, None);
        let b = builder.load(&a_ptr.get(1), &Time::zero(), None, None);
        let c = builder.and(&a, &b);
        builder.store(&a_0_trace, c, &clk.advance(), None, None, None);

        let a_final = builder.alloc_public::<U32Register>();

        builder.free(&a_0, a_final, &Time::constant(num_rows));
        builder.set_to_expression_last_row(&a_final, c.expr());

        for (i, a) in a_init.iter().enumerate().skip(1) {
            builder.api.free(&a_ptr.get(i), a, &Time::zero());
        }

        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        let mut rng = rand::thread_rng();

        let a_val = (0..a_init.len())
            .map(|_| u32_to_le_field_bytes(rng.gen::<u32>()))
            .collect::<Vec<_>>();
        writer.write_array(&a_init, a_val, 0);
        writer.write_global_instructions(&stark.air_data);
        for i in 0..num_rows {
            writer.write_row_instructions(&stark.air_data, i);
        }

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);

        let data = recursive_builder.build::<Config>();

        let mut pw = PartialWitness::new();

        pw.set_target_arr(&public_input, &public);
        stark.set_proof_target(&mut pw, &proof_target, proof);

        let rec_proof = data.prove(pw).unwrap();
        data.verify(rec_proof).unwrap();

        timing.print();
    }
}
