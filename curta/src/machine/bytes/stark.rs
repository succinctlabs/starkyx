use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger::Challenger;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use super::air::ByteParameters;
use super::proof::{ByteAirCommitment, ByteStarkProof};
use crate::chip::trace::data::AirTraceData;
use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
use crate::chip::uint::bytes::lookup_table::multiplicity_data::ByteMultiplicityData;
use crate::chip::uint::bytes::lookup_table::table::ByteLogLookupTable;
use crate::chip::{AirParameters, Chip};
use crate::machine::bytes::builder::NUM_LOOKUP_ROWS;
use crate::maybe_rayon::*;
use crate::plonky2::stark::config::{CurtaConfig, StarkyConfig};
use crate::plonky2::stark::proof::StarkProof;
use crate::plonky2::stark::prover::StarkyProver;
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

impl<L: AirParameters, C: CurtaConfig<D, F = L::Field>, const D: usize> ByteStark<L, C, D>
where
    L::Field: RichField + Extendable<D>,
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
    }

    fn generate_trace(
        &self,
        execusion_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
        challenger: &mut Challenger<L::Field, C::Hasher>,
        timing: &mut TimingTree,
    ) -> ByteAirCommitment<L::Field, C, D> {
        // Absorve public values into the challenger.
        challenger.observe_elements(public_values);

        // Generate execusion traces.
        let (main_writer, lookup_writer) =
            self.generate_execusion_traces(execusion_trace, public_values);

        // Commit to execusion traces
        let main_execusion_commitment = timed!(
            timing,
            "Commit to execusion trace",
            self.config
                .commit(&main_writer.read_trace().unwrap(), timing)
        );

        let lookup_execusion_commitment = timed!(
            timing,
            "Commit to lookup execusion trace",
            self.lookup_config
                .commit(&lookup_writer.read_trace().unwrap(), timing)
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

        // Destruct writers.
        let InnerWriterData {
            trace: main_trace,
            challenges: main_challenges,
            ..
        } = main_writer.into_inner().unwrap();
        let InnerWriterData {
            trace: lookup_trace,
            public: lookup_public,
            global: lookup_global,
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

        // Return the air commitment.
        ByteAirCommitment {
            main_trace_commitments: vec![main_execusion_commitment, main_extended_commitment],
            lookup_trace_commitments: vec![lookup_execusion_commitment, lookup_extended_commitment],
            public_inputs: lookup_public,
            global_values: lookup_global,
            challenges: main_challenges,
        }
    }

    pub fn prove(
        &self,
        execusion_trace: &AirTrace<L::Field>,
        public_values: &[L::Field],
        timing: &mut TimingTree,
    ) -> ByteStarkProof<L::Field, C, D> {
        // Initialize challenger.
        let mut challenger = Challenger::new();

        // Generate stark commitment.
        let (main_air_commitment, lookup_air_commitment) = timed!(
            timing,
            "Generate stark trace",
            self.generate_trace(execusion_trace, public_values, &mut challenger, timing)
                .air_commitments()
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
            )
            .unwrap()
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
            )
            .unwrap()
        );

        // Return the proof.
        ByteStarkProof {
            main_proof: main_proof.air_proof,
            lookup_proof: lookup_proof.air_proof,
            global_values: lookup_proof.global_values,
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
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
        let mut challenger = Challenger::new();
        let _ = stark.generate_trace(&trace, &public, &mut challenger, &mut timing);

        timing.print();
    }
}
