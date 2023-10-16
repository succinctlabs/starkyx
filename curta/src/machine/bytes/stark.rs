use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger::{Challenger, RecursiveChallenger};
use plonky2::iop::target::Target;
use plonky2::iop::witness::WitnessWrite;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use super::air::ByteParameters;
use super::proof::{
    ByteStarkChallenges, ByteStarkChallengesTarget, ByteStarkProof, ByteStarkProofTarget,
};
use crate::chip::trace::data::AirTraceData;
use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
use crate::chip::uint::bytes::lookup_table::multiplicity_data::ByteMultiplicityData;
use crate::chip::uint::bytes::lookup_table::table::ByteLogLookupTable;
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

pub struct ByteStark<L: AirParameters, C, const D: usize> {
    pub(crate) config: StarkyConfig<C, D>,
    pub(crate) stark: Starky<Chip<L>>,
    pub(crate) air_data: AirTraceData<L>,
    pub(crate) multiplicity_data: ByteMultiplicityData,
    pub(crate) lookup_config: StarkyConfig<C, D>,
    pub(crate) lookup_stark: Starky<Chip<ByteParameters<L::Field, L::CubicParams>>>,
    pub(crate) lookup_air_data: AirTraceData<ByteParameters<L::Field, L::CubicParams>>,
    pub(crate) lookup_table: ByteLogLookupTable<L::Field, L::CubicParams>,
}

impl<L: AirParameters, C, const D: usize> ByteStark<L, C, D>
where
    L::Field: RichField + Extendable<D>,
    C: CurtaConfig<D, F = L::Field, FE = <L::Field as Extendable<D>>::Extension>,
    Chip<L>: Plonky2Air<L::Field, D>,
{
    fn generate_execusion_traces(
        &self,
        execusion_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
    ) -> (TraceWriter<L::Field>, TraceWriter<L::Field>) {
        // Initialize writers.
        let main_writer = TraceWriter::new(&self.air_data, execusion_trace.height());
        let lookup_writer = TraceWriter::new(&self.lookup_air_data, NUM_LOOKUP_ROWS);

        // Insert execusion trace and into main writer.
        let execusion_trace_length = self.stark.air.execution_trace_length;
        main_writer
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .zip(execusion_trace.rows_par())
            .for_each(|(row, execusion_row)| {
                row[0..execusion_trace_length]
                    .copy_from_slice(&execusion_row[0..execusion_trace_length]);
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
        self.air_data.write_extended_trace(main_writer);

        // Update global values
        lookup_writer
            .global
            .write()
            .unwrap()
            .copy_from_slice(&main_writer.global.read().unwrap());

        // Write the extended trace values
        self.lookup_air_data.write_extended_trace(lookup_writer);

        // Update global values
        main_writer
            .global
            .write()
            .unwrap()
            .copy_from_slice(&lookup_writer.global.read().unwrap());
    }

    fn generate_trace(
        &self,
        execusion_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
        challenger: &mut Challenger<L::Field, C::Hasher>,
        timing: &mut TimingTree,
    ) -> (AirCommitment<L::Field, C, D>, AirCommitment<L::Field, C, D>) {
        // Absorve public values into the challenger.
        challenger.observe_elements(public_values);

        // Generate execusion traces.
        let (main_writer, lookup_writer) =
            self.generate_execusion_traces(execusion_trace, public_values);

        let main_execusion_trace_values = main_writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[0..self.stark.air.execution_trace_length].to_vec())
            .collect::<Vec<_>>();
        let main_execusion_trace = AirTrace {
            values: main_execusion_trace_values,
            width: self.stark.air.execution_trace_length,
        };

        let lookup_execusion_trace_values = lookup_writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[0..self.lookup_stark.air.execution_trace_length].to_vec())
            .collect::<Vec<_>>();

        let lookup_execusion_trace = AirTrace {
            values: lookup_execusion_trace_values,
            width: self.lookup_stark.air.execution_trace_length,
        };

        // Commit to execusion traces
        let main_execusion_commitment = timed!(
            timing,
            "Commit to execusion trace",
            self.config.commit(&main_execusion_trace, timing)
        );

        let lookup_execusion_commitment = timed!(
            timing,
            "Commit to lookup execusion trace",
            self.lookup_config.commit(&lookup_execusion_trace, timing)
        );

        // Absorve the trace commitments into the challenger.
        challenger.observe_cap(&main_execusion_commitment.merkle_tree.cap);
        challenger.observe_cap(&lookup_execusion_commitment.merkle_tree.cap);

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
            .flat_map(|row| row[self.lookup_stark.air.execution_trace_length..].to_vec())
            .collect::<Vec<_>>();
        let lookup_extended_trace = AirTrace {
            values: lookup_extended_trace_values,
            width: ByteParameters::<L::Field, L::CubicParams>::num_columns()
                - self.lookup_stark.air.execution_trace_length,
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
                trace_commitments: vec![main_execusion_commitment, main_extended_commitment],
                public_inputs: main_public,
                global_values: main_global,
                challenges: main_challenges,
            },
            AirCommitment {
                trace_commitments: vec![lookup_execusion_commitment, lookup_extended_commitment],
                public_inputs: lookup_public,
                global_values: lookup_global,
                challenges: global_challenges,
            },
        )
    }

    pub fn prove(
        &self,
        execusion_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
        timing: &mut TimingTree,
    ) -> Result<ByteStarkProof<L::Field, C, D>> {
        // Initialize challenger.
        let mut challenger = Challenger::new();

        // Generate stark commitment.
        let (main_air_commitment, lookup_air_commitment) = timed!(
            timing,
            "Generate stark trace",
            self.generate_trace(execusion_trace, public_values, &mut challenger, timing)
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

        // Observe execusion trace commitments.
        challenger.observe_cap(&proof.main_proof.trace_caps[0]);
        challenger.observe_cap(&proof.lookup_proof.trace_caps[0]);

        // Get challenges.
        let challenges = challenger.get_n_challenges(self.stark.air.num_challenges);

        // Observe global values.
        challenger.observe_elements(&proof.global_values);
        // Observe extended trace commitments.
        challenger.observe_cap(&proof.main_proof.trace_caps[1]);
        challenger.observe_cap(&proof.lookup_proof.trace_caps[1]);

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

        StarkyVerifier::verify_with_challenges(
            &self.config,
            &self.stark,
            main_proof,
            public_values,
            &global_values,
            main_challenges,
        )?;
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

        // Observe execusion trace commitments.
        challenger.observe_cap(&proof.main_proof.trace_caps[0]);
        challenger.observe_cap(&proof.lookup_proof.trace_caps[0]);

        // Get challenges.
        let challenges = challenger.get_n_challenges(builder, self.stark.air.num_challenges);

        // Observe global values.
        challenger.observe_elements(&proof.global_values);
        // Observe extended trace commitments.
        challenger.observe_cap(&proof.main_proof.trace_caps[1]);
        challenger.observe_cap(&proof.lookup_proof.trace_caps[1]);

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
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use rand::Rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::trace::writer::InnerWriterData;
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::uint::register::U32Register;
    use crate::chip::uint::util::u32_to_le_field_bytes;
    use crate::machine::bytes::builder::BytesBuilder;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::plonky2::stark::config::CurtaPoseidonGoldilocksConfig;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ByteTest;

    impl AirParameters for ByteTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 88;
        const EXTENDED_COLUMNS: usize = 63;
    }

    #[test]
    fn test_byte_multi_stark() {
        type L = ByteTest;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_byte_multi_stark", log::Level::Debug);

        let mut builder = BytesBuilder::<L>::new();

        let a = builder.api.alloc::<U32Register>();
        let b = builder.api.alloc::<U32Register>();

        let num_ops = 10;
        for _ in 0..num_ops {
            let _ = builder.bitwise_and(&a, &b);
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
}
