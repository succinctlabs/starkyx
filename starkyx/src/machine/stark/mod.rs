use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger::{Challenger, RecursiveChallenger};
use plonky2::iop::target::Target;
use plonky2::iop::witness::WitnessWrite;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::chip::table::log_derivative::entry::LogEntry;
use crate::chip::table::lookup::table::LookupTable;
use crate::chip::table::lookup::values::LookupValues;
use crate::chip::trace::data::AirTraceData;
use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::proof::{
    StarkProof, StarkProofChallenges, StarkProofChallengesTarget, StarkProofTarget,
};
use crate::plonky2::stark::prover::{AirCommitment, StarkyProver};
use crate::plonky2::stark::verifier::{
    add_virtual_air_proof, set_air_proof_target, StarkyVerifier,
};
use crate::plonky2::stark::Starky;
use crate::plonky2::Plonky2Air;
use crate::trace::AirTrace;

pub mod builder;

pub struct Stark<L: AirParameters, C, const D: usize> {
    pub config: StarkyConfig<C, D>,
    pub stark: Starky<Chip<L>>,
    pub air_data: AirTraceData<L>,
}

impl<L: AirParameters, C, const D: usize> Stark<L, C, D>
where
    L::Field: RichField + Extendable<D>,
    C: CurtaConfig<D, F = L::Field, FE = <L::Field as Extendable<D>>::Extension>,
    Chip<L>: Plonky2Air<L::Field, D>,
{
    pub const fn ineer_stark(&self) -> &Starky<Chip<L>> {
        &self.stark
    }

    pub const fn config(&self) -> &StarkyConfig<C, D> {
        &self.config
    }

    #[inline]
    pub fn range_fn(element: L::Field) -> usize {
        element.as_canonical_u64() as usize
    }

    fn generate_execution_trace(
        &self,
        execution_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
    ) -> TraceWriter<L::Field> {
        let num_rows = execution_trace.height();
        // Initialize writers.
        let writer = TraceWriter::new(&self.air_data, num_rows);

        // Insert execution trace and into writer.
        let execution_trace_length = self.stark.air.execution_trace_length;
        writer
            .write_trace()
            .unwrap()
            .rows_par_mut()
            .zip(execution_trace.rows_par())
            .for_each(|(row, execution_row)| {
                row[0..execution_trace_length]
                    .copy_from_slice(&execution_row[0..execution_trace_length]);
            });
        // Insert public inputs into writer.
        writer.public_mut().unwrap().copy_from_slice(public_values);

        // Write the range check table and multiplicitiies
        if let Some((LookupTable::Element(table), LookupValues::Element(values))) =
            &self.air_data.range_data
        {
            assert_eq!(table.table.len(), 1);
            let table_column = table.table[0];
            for i in 0..num_rows {
                writer.write(&table_column, &L::Field::from_canonical_usize(i), i);
            }

            writer.write_multiplicities_from_fn(
                num_rows,
                table,
                Self::range_fn,
                &values
                    .trace_values
                    .iter()
                    .map(LogEntry::value)
                    .copied()
                    .collect::<Vec<_>>(),
                &values
                    .public_values
                    .iter()
                    .map(LogEntry::value)
                    .copied()
                    .collect::<Vec<_>>(),
            );
        }

        writer
    }

    fn generate_extended_trace(&self, writer: &TraceWriter<L::Field>) {
        self.air_data.write_extended_trace(writer);
    }

    fn generate_trace(
        &self,
        execution_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
        challenger: &mut Challenger<L::Field, C::Hasher>,
        timing: &mut TimingTree,
    ) -> AirCommitment<L::Field, C, D> {
        // Absorve public values into the challenger.
        challenger.observe_elements(public_values);

        // Generate execution trace.
        let writer = self.generate_execution_trace(execution_trace, public_values);

        let execution_trace_values = writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[0..self.stark.air.execution_trace_length].to_vec())
            .collect::<Vec<_>>();
        let execution_trace = AirTrace {
            values: execution_trace_values,
            width: self.stark.air.execution_trace_length,
        };

        // Commit to execution traces
        let execution_commitment = timed!(
            timing,
            "Commit to execution trace",
            self.config.commit(&execution_trace, timing)
        );

        // Absorve the trace commitments into the challenger.
        challenger.observe_cap(&execution_commitment.merkle_tree.cap);

        // Get random AIR challenges.
        let challenges = challenger.get_n_challenges(self.stark.air.num_challenges);
        // Save challenges to the writer.
        writer
            .challenges
            .write()
            .unwrap()
            .extend_from_slice(&challenges);

        // Generate extended trace.
        self.generate_extended_trace(&writer);

        let InnerWriterData {
            trace,
            public,
            global,
            challenges,
            ..
        } = writer.into_inner().unwrap();

        // Commit to extended traces.
        let extended_trace_values = trace
            .rows_par()
            .flat_map(|row| row[self.stark.air.execution_trace_length..].to_vec())
            .collect::<Vec<_>>();
        let extended_trace = AirTrace {
            values: extended_trace_values,
            width: L::num_columns() - self.stark.air.execution_trace_length,
        };
        let extended_commitment = timed!(
            timing,
            "Commit to extended trace",
            self.config.commit(&extended_trace, timing)
        );

        // Obsderve global values.
        challenger.observe_elements(&global);
        // Observe extended trace commitments.
        challenger.observe_cap(&extended_commitment.merkle_tree.cap);

        // Return the air commitment.
        AirCommitment {
            trace_commitments: vec![execution_commitment, extended_commitment],
            public_inputs: public,
            global_values: global,
            challenges,
        }
    }

    pub fn prove(
        &self,
        execution_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
        timing: &mut TimingTree,
    ) -> Result<StarkProof<L::Field, C, D>> {
        // Initialize challenger.
        let mut challenger = Challenger::new();

        // Generate stark commitment.
        let air_commitment = timed!(
            timing,
            "Generate stark trace",
            self.generate_trace(execution_trace, public_values, &mut challenger, timing)
        );

        // Generate individual stark proofs.
        let proof = timed!(
            timing,
            "Generate main proof",
            StarkyProver::prove_with_trace(
                &self.config,
                &self.stark,
                air_commitment,
                &mut challenger,
                &mut TimingTree::default(),
            )?
        );

        // Return the proof.
        Ok(proof)
    }

    pub fn get_challenges(
        &self,
        proof: &StarkProof<L::Field, C, D>,
        public_values: &[L::Field],
    ) -> StarkProofChallenges<L::Field, D> {
        // Initialize challenger.
        let mut challenger = Challenger::<L::Field, C::Hasher>::new();

        // Observe public values.
        challenger.observe_elements(public_values);

        // Observe execution trace commitments.
        challenger.observe_cap(&proof.air_proof.trace_caps[0]);

        // Get challenges.
        let challenges = challenger.get_n_challenges(self.stark.air.num_challenges);

        // Observe global values.
        challenger.observe_elements(&proof.global_values);
        // Observe extended trace commitments.
        challenger.observe_cap(&proof.air_proof.trace_caps[1]);

        // Get all challenges.
        proof.air_proof.get_iop_challenges(
            &self.config,
            self.config.degree_bits,
            challenges.clone(),
            &mut challenger,
        )
    }

    pub fn verify(
        &self,
        proof: StarkProof<L::Field, C, D>,
        public_values: &[L::Field],
    ) -> Result<()> {
        let challenges = self.get_challenges(&proof, public_values);

        let StarkProof {
            air_proof,
            global_values,
        } = proof;

        StarkyVerifier::verify_with_challenges(
            &self.config,
            &self.stark,
            air_proof,
            public_values,
            &global_values,
            challenges,
        )
    }

    pub fn add_virtual_proof_with_pis_target(
        &self,
        builder: &mut CircuitBuilder<L::Field, D>,
    ) -> (StarkProofTarget<D>, Vec<Target>) {
        let air_proof = add_virtual_air_proof(builder, &self.stark, &self.config);

        let num_global_values = self.stark.air.num_global_values;
        let global_values = builder.add_virtual_targets(num_global_values);
        let public_inputs = builder.add_virtual_targets(self.stark.air.num_public_values);

        (
            StarkProofTarget {
                air_proof,
                global_values,
            },
            public_inputs,
        )
    }

    pub fn get_challenges_target(
        &self,
        builder: &mut CircuitBuilder<L::Field, D>,
        proof: &StarkProofTarget<D>,
        public_values: &[Target],
    ) -> StarkProofChallengesTarget<D> {
        // Initialize challenger.
        let mut challenger = RecursiveChallenger::<L::Field, C::InnerHasher, D>::new(builder);

        // Observe public values.
        challenger.observe_elements(public_values);

        // Observe execution trace commitments.
        challenger.observe_cap(&proof.air_proof.trace_caps[0]);

        // Get challenges.
        let challenges = challenger.get_n_challenges(builder, self.stark.air.num_challenges);

        // Observe global values.
        challenger.observe_elements(&proof.global_values);
        // Observe extended trace commitments.
        challenger.observe_cap(&proof.air_proof.trace_caps[1]);

        // Get all challenges.
        proof.air_proof.get_iop_challenges_target(
            builder,
            &self.config,
            challenges.clone(),
            &mut challenger,
        )
    }

    pub fn verify_circuit(
        &self,
        builder: &mut CircuitBuilder<L::Field, D>,
        proof: &StarkProofTarget<D>,
        public_values: &[Target],
    ) {
        let challenges = self.get_challenges_target(builder, proof, public_values);
        let StarkProofTarget {
            air_proof,
            global_values,
        } = proof;

        StarkyVerifier::verify_with_challenges_circuit(
            builder,
            &self.config,
            &self.stark,
            air_proof,
            public_values,
            global_values,
            challenges,
        );
    }

    pub fn set_proof_target<W: WitnessWrite<L::Field>>(
        &self,
        witness: &mut W,
        proof_tagret: &StarkProofTarget<D>,
        proof: StarkProof<L::Field, C, D>,
    ) {
        let StarkProofTarget {
            air_proof,
            global_values,
        } = proof_tagret;

        set_air_proof_target(witness, air_proof, &proof.air_proof);

        witness.set_target_arr(global_values, &proof.global_values);
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::field::instruction::FpInstruction;
    use crate::chip::field::parameters::tests::Fp25519;
    use crate::chip::field::parameters::FieldParameters;
    use crate::chip::field::register::FieldRegister;
    use crate::chip::trace::writer::data::AirWriterData;
    use crate::chip::trace::writer::AirWriter;
    use crate::machine::builder::Builder;
    use crate::machine::stark::builder::StarkBuilder;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::plonky2::stark::config::CurtaPoseidonGoldilocksConfig;
    use crate::polynomial::Polynomial;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RangeTest;

    impl AirParameters for RangeTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = FpInstruction<Fp25519>;

        const NUM_ARITHMETIC_COLUMNS: usize = 124;
        const NUM_FREE_COLUMNS: usize = 3;
        const EXTENDED_COLUMNS: usize = 198;
    }

    #[test]
    fn test_fp_single_stark() {
        type L = RangeTest;
        type F = GoldilocksField;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_byte_multi_stark", log::Level::Debug);

        let mut builder = StarkBuilder::<L>::new();

        let a = builder.alloc::<FieldRegister<Fp25519>>();
        let b = builder.alloc::<FieldRegister<Fp25519>>();

        let num_rows = 1 << 16;
        let stark = builder.build::<C, 2>(num_rows);

        let mut writer_data = AirWriterData::new(&stark.air_data, num_rows);

        let p = Fp25519::modulus();
        let air_data = &stark.air_data;
        air_data.write_global_instructions(&mut writer_data.public_writer());

        let k = 1 << 0;
        writer_data.chunks(k).for_each(|mut chunk| {
            let mut rng = rand::thread_rng();
            for i in 0..k {
                let mut writer = chunk.row_writer(i);
                let a_int = rng.gen_biguint(256) % &p;
                let b_int = rng.gen_biguint(256) % &p;
                let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, 16);
                let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, 16);
                writer.write(&a, &p_a);
                writer.write(&b, &p_b);
                air_data.write_trace_instructions(&mut writer);
            }
        });

        let (trace, public) = (writer_data.trace, writer_data.public);

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
