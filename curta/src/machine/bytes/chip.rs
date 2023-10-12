#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::fri::oracle::PolynomialBatch;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::GenericConfig;
    use plonky2::util::timing::TimingTree;
    use rand::Rng;
    use serde::{Deserialize, Serialize};

    use crate::chip::builder::shared_memory::SharedMemory;
    use crate::chip::builder::AirBuilder;
    use crate::chip::trace::writer::TraceWriter;
    use crate::chip::uint::operations::instruction::U32Instruction;
    use crate::chip::uint::register::U32Register;
    use crate::chip::uint::util::u32_to_le_field_bytes;
    use crate::chip::AirParameters;
    use crate::machine::bytes::air::ByteParameters;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::maybe_rayon::*;
    use crate::plonky2::challenger::{Plonky2Challenger, Plonky2RecursiveChallenger};
    use crate::plonky2::stark::config::{
        CurtaConfig, CurtaPoseidonGoldilocksConfig, PoseidonGoldilocksStarkConfig,
    };
    use crate::plonky2::stark::gadget::StarkGadget;
    use crate::plonky2::stark::prover::{AirCommitment, StarkyProver};
    use crate::plonky2::stark::verifier::{set_stark_proof_target, StarkyVerifier};
    use crate::plonky2::stark::Starky;
    use crate::trace::AirTrace;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ByteTest;

    impl AirParameters for ByteTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Instruction;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 88;
        const EXTENDED_COLUMNS: usize = 63;
    }

    #[test]
    fn test_mult_table_byte_ops() {
        type C = CurtaPoseidonGoldilocksConfig;
        type Hasher = <C as CurtaConfig<2>>::Hasher;
        type Config = <C as CurtaConfig<2>>::GenericConfig;
        type F = GoldilocksField;
        const D: usize = 2;

        let _ = env_logger::builder().is_test(true).try_init();

        let shared_memory = SharedMemory::new();

        let mut table_builder = AirBuilder::<ByteParameters>::init(shared_memory.clone());
        let mut builder = AirBuilder::<ByteTest>::init(shared_memory);

        let mut table = table_builder.new_byte_lookup_table();

        let mut operations = builder.byte_operations();

        let a = builder.alloc::<U32Register>();
        let b = builder.alloc::<U32Register>();

        let num_ops = 10;
        for _ in 0..num_ops {
            let _ = builder.bitwise_and(&a, &b, &mut operations);
        }

        let byte_data = builder.register_byte_lookup(&mut table, operations);

        table_builder.constraint_byte_lookup_table(&table);

        let (air, trace_data) = builder.build();
        let (table_air, table_trace_data) = table_builder.build();

        let stark = Starky::new(air);
        let table_stark = Starky::new(table_air);

        let num_rows = 1 << 5;
        let config = PoseidonGoldilocksStarkConfig::standard_fast_config(num_rows);
        let table_config = PoseidonGoldilocksStarkConfig::standard_fast_config(1 << 16);

        let writer = TraceWriter::new(&trace_data, num_rows);
        let byte_writer = TraceWriter::new(&table_trace_data, 1 << 16);

        let mut challenger = Plonky2Challenger::<F, Hasher>::new();

        table.write_table_entries(&byte_writer);
        for i in 0..(1 << 16) {
            byte_writer.write_row_instructions(&table_trace_data, i);
        }

        let mut rng = rand::thread_rng();
        for i in 0..num_rows {
            let a_val = rng.gen::<u32>();
            let b_val = rng.gen::<u32>();
            writer.write(&a, &u32_to_le_field_bytes(a_val), i);
            writer.write(&b, &u32_to_le_field_bytes(b_val), i);
            writer.write_row_instructions(&trace_data, i);
        }
        let multiplicities = byte_data.get_multiplicities(&writer);
        byte_writer.write_lookup_multiplicities(table.multiplicities(), &[multiplicities]);

        // Get the trace commitments.
        let execution_trace_values = writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[..stark.air.execution_trace_length].to_vec())
            .collect::<Vec<_>>();
        let execusion_trace = AirTrace {
            values: execution_trace_values,
            width: stark.air.execution_trace_length,
        };

        let trace_cols = execusion_trace
            .as_columns()
            .into_par_iter()
            .map(PolynomialValues::from)
            .collect::<Vec<_>>();
        let rate_bits = config.fri_config.rate_bits;
        let cap_height = config.fri_config.cap_height;
        let commitment = PolynomialBatch::<F, Config, D>::from_values(
            trace_cols,
            rate_bits,
            false,
            cap_height,
            &mut TimingTree::default(),
            None,
        );

        let byte_execution_trace_values = byte_writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[..table_stark.air.execution_trace_length].to_vec())
            .collect::<Vec<_>>();
        let byte_execusion_trace = AirTrace {
            values: byte_execution_trace_values,
            width: table_stark.air.execution_trace_length,
        };

        let trace_cols = byte_execusion_trace
            .as_columns()
            .into_par_iter()
            .map(PolynomialValues::from)
            .collect::<Vec<_>>();
        let rate_bits = config.fri_config.rate_bits;
        let cap_height = config.fri_config.cap_height;
        let table_commitment = PolynomialBatch::<F, Config, D>::from_values(
            trace_cols,
            rate_bits,
            false,
            cap_height,
            &mut TimingTree::default(),
            None,
        );

        // challenger.0.observe_cap(&commitment.merkle_tree.cap);
        // challenger.0.observe_cap(&table_commitment.merkle_tree.cap);

        let challenges = challenger.0.get_n_challenges(stark.air.num_challenges);

        // write challenges to both traces.
        let mut challenges_write = writer.challenges.write().unwrap();
        challenges_write.extend_from_slice(&challenges);
        drop(challenges_write);

        let mut challenges_write = byte_writer.challenges.write().unwrap();
        challenges_write.extend_from_slice(&challenges);
        drop(challenges_write);

        // Write the extended trace values.
        trace_data.write_extended_trace(&writer);

        // Update global values
        let global = writer.global.read().unwrap();
        let mut global_write = byte_writer.global.write().unwrap();
        global_write.copy_from_slice(&global);
        drop(global_write);

        // Write the extended trace values
        table_trace_data.write_extended_trace(&byte_writer);

        let public_inputs = writer.public().unwrap().clone();
        let global_values = byte_writer.global().unwrap().clone();

        // Make air commitments.
        let extended_trace_values = writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[stark.air.execution_trace_length..].to_vec())
            .collect::<Vec<_>>();

        let extended_trace = AirTrace {
            values: extended_trace_values,
            width: ByteTest::num_columns() - stark.air.execution_trace_length,
        };
        let trace_cols = extended_trace
            .as_columns()
            .into_par_iter()
            .map(PolynomialValues::from)
            .collect::<Vec<_>>();
        let extended_commitment = PolynomialBatch::<F, Config, D>::from_values(
            trace_cols,
            rate_bits,
            false,
            cap_height,
            &mut TimingTree::default(),
            None,
        );

        let extended_table_trace_values = byte_writer
            .read_trace()
            .unwrap()
            .rows_par()
            .flat_map(|row| row[table_stark.air.execution_trace_length..].to_vec())
            .collect::<Vec<_>>();

        let table_extended_trace = AirTrace {
            values: extended_table_trace_values,
            width: ByteParameters::num_columns() - table_stark.air.execution_trace_length,
        };
        let trace_cols = table_extended_trace
            .as_columns()
            .into_par_iter()
            .map(PolynomialValues::from)
            .collect::<Vec<_>>();
        let table_extended_commitment = PolynomialBatch::<F, Config, D>::from_values(
            trace_cols,
            rate_bits,
            false,
            cap_height,
            &mut TimingTree::default(),
            None,
        );

        let air_commitment = AirCommitment::<F, C, D> {
            trace_commitments: vec![commitment, extended_commitment],
            public_inputs: public_inputs.clone(),
            global_values: global_values.clone(),
            challenges: challenges.clone(),
        };

        let table_air_commitment = AirCommitment::<F, C, D> {
            trace_commitments: vec![table_commitment, table_extended_commitment],
            public_inputs: public_inputs.clone(),
            global_values,
            challenges: challenges.clone(),
        };

        let mut verfier_challenger = challenger.clone();

        let proof = StarkyProver::prove_with_trace(
            &config,
            &stark,
            air_commitment,
            &mut challenger,
            &mut TimingTree::default(),
        )
        .unwrap();

        let table_proof = StarkyProver::prove_with_trace(
            &table_config,
            &table_stark,
            table_air_commitment,
            &mut challenger,
            &mut TimingTree::default(),
        )
        .unwrap();

        let stark_challenges = proof.get_iop_challenges(
            &config,
            config.degree_bits,
            challenges.clone(),
            &mut verfier_challenger,
        );

        let table_challenges = table_proof.get_iop_challenges(
            &table_config,
            table_config.degree_bits,
            challenges.clone(),
            &mut verfier_challenger,
        );

        StarkyVerifier::verify_with_challenges(
            &config,
            &stark,
            proof.clone(),
            &public_inputs,
            stark_challenges,
        )
        .unwrap();

        StarkyVerifier::verify_with_challenges(
            &table_config,
            &table_stark,
            table_proof.clone(),
            &public_inputs,
            table_challenges,
        )
        .unwrap();

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config_rec);
        let mut recursive_challenger =
            Plonky2RecursiveChallenger::<F, <Config as GenericConfig<2>>::InnerHasher, 2>::new(
                &mut builder,
            );
        let challenges_target = recursive_challenger
            .0
            .get_n_challenges(&mut builder, stark.air.num_challenges);
        let public_inputs_target = builder.constants(&public_inputs);
        let virtual_proof = builder.add_virtual_stark_proof(&stark, &config);
        let stark_challenges_target = virtual_proof.get_iop_challenges_target(
            &mut builder,
            &config,
            challenges_target.clone(),
            &mut recursive_challenger,
        );
        StarkyVerifier::verify_with_challenges_circuit(
            &mut builder,
            &config,
            &stark,
            &virtual_proof,
            &public_inputs_target,
            stark_challenges_target,
        );

        let table_virtual_proof = builder.add_virtual_stark_proof(&table_stark, &table_config);
        let table_challenges_target = table_virtual_proof.get_iop_challenges_target(
            &mut builder,
            &table_config,
            challenges_target,
            &mut recursive_challenger,
        );
        StarkyVerifier::verify_with_challenges_circuit(
            &mut builder,
            &table_config,
            &table_stark,
            &table_virtual_proof,
            &public_inputs_target,
            table_challenges_target,
        );

        let data = builder.build::<Config>();
        let mut pw = PartialWitness::new();
        set_stark_proof_target(&mut pw, &virtual_proof, &proof);
        set_stark_proof_target(&mut pw, &table_virtual_proof, &table_proof);

        let recursive_proof = data.prove(pw).unwrap();
        data.verify(recursive_proof).unwrap();
    }
}
