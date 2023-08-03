// //! U32

// pub mod channel;
// pub mod opcode;
// pub mod operation;
// pub mod write_operation;

// #[cfg(test)]
// mod tests {
//     use plonky2::timed;
//     use plonky2::util::timing::TimingTree;
//     use rand::{thread_rng, Rng};

//     use super::operation::U32Operation;
//     pub use crate::chip::builder::tests::*;
//     use crate::chip::builder::AirBuilder;
//     use crate::chip::register::cubic::CubicRegister;
//     use crate::chip::u32::opcode::{OPCODE_AND, OPCODE_XOR};
//     use crate::chip::u32::write_operation::U32OperationWrite;
//     use crate::chip::AirParameters;
//     use crate::math::prelude::*;

//     #[derive(Debug, Clone)]
//     pub struct U32OperationTest;

//     impl const AirParameters for U32OperationTest {
//         type Field = GoldilocksField;
//         type CubicParams = GoldilocksCubicParameters;

//         type Instruction = U32Operation;

//         const NUM_FREE_COLUMNS: usize = 104;
//         const EXTENDED_COLUMNS: usize = 18;

//         fn num_rows_bits() -> usize {
//             14
//         }
//     }

//     #[test]
//     fn test_xor_opcode() {
//         type F = GoldilocksField;
//         type L = U32OperationTest;
//         type SC = PoseidonGoldilocksStarkConfig;

//         let mut builder = AirBuilder::<L>::new();

//         let a = builder.alloc::<ElementRegister>();
//         let b = builder.alloc::<ElementRegister>();
//         let opcode_dst = builder.u32_opcode(OPCODE_XOR, a, b);

//         let a_table = builder.alloc::<ElementRegister>();
//         let b_table = builder.alloc::<ElementRegister>();
//         let opcode_table = builder.u32_opcode(OPCODE_XOR, a_table, b_table);
//         let op = builder.u32_operation_from_opcode(&opcode_table);
//         let op_write = U32OperationWrite::new(opcode_table.clone(), op);

//         // Set up the bus
//         let mut bus = builder.new_bus();
//         let challenges = builder.alloc_challenge_array::<CubicRegister>(4);

//         let mut write_channel = builder.new_u32_channel(&challenges, &mut bus);
//         let mut read_channel = builder.new_u32_channel(&challenges, &mut bus);

//         builder.input_to_u32_channel(&opcode_dst, &mut read_channel);
//         builder.output_from_u32_channel(&opcode_table, &mut write_channel);

//         // constrain the bus
//         builder.constrain_bus(bus);

//         let air_1 = builder.build();
//         let air_2 = air_1.clone();

//         let generator_1 = ArithmeticGenerator::<L>::new(&air_1);
//         let generator_2 = ArithmeticGenerator::<L>::new(&air_2);

//         let writer_1 = generator_1.new_writer();
//         let writer_2 = generator_2.new_writer();

//         let mut rng = thread_rng();
//         for i in 0..L::num_rows() {
//             let a_val = F::from_canonical_u32(rng.gen::<u32>());
//             let b_val = F::from_canonical_u32(rng.gen::<u32>());

//             writer_1.write(&a, &a_val, L::num_rows() - 1 - i);
//             writer_1.write(&b, &b_val, L::num_rows() - 1 - i);

//             writer_1.write(&a_table, &a_val, i);
//             writer_1.write(&b_table, &b_val, i);

//             writer_1.write_instruction(&opcode_dst, L::num_rows() - 1 - i);
//             writer_1.write_instruction(&op_write, i);

//             writer_2.write(&a, &a_val, L::num_rows() - 1 - i);
//             writer_2.write(&b, &b_val, L::num_rows() - 1 - i);

//             writer_2.write(&a_table, &a_val, i);
//             writer_2.write(&b_table, &b_val, i);

//             writer_2.write_instruction(&opcode_dst, L::num_rows() - 1 - i);
//             writer_2.write_instruction(&op_write, i);
//         }

//         let stark_1 = Starky::<_, { L::num_columns() }>::new(air_1);
//         let stark_2 = Starky::<_, { L::num_columns() }>::new(air_2);
//         let config_1 = SC::standard_fast_config(L::num_rows());
//         let config_2 = SC::standard_fast_config(L::num_rows());

//         // Generate proof and verify as a stark
//         // test_starky(&stark, &config, &generator, &[]);

//         // Test the recursive proof.
//         rayon::join(
//             || test_recursive_starky(stark_1, config_1, generator_1, &[]),
//             || test_recursive_starky(stark_2, config_2, generator_2, &[]),
//         );
//     }

//     #[test]
//     fn test_and_opcode() {
//         type F = GoldilocksField;
//         type L = U32OperationTest;
//         type SC = PoseidonGoldilocksStarkConfig;

//         let _ = env_logger::builder().is_test(true).try_init();
//         let mut timing = TimingTree::new("And opcode", log::Level::Debug);

//         let mut builder = AirBuilder::<L>::new();

//         let a = builder.alloc::<ElementRegister>();
//         let b = builder.alloc::<ElementRegister>();
//         let opcode_dst = builder.u32_opcode(OPCODE_AND, a, b);

//         let a_table = builder.alloc::<ElementRegister>();
//         let b_table = builder.alloc::<ElementRegister>();
//         let opcode_table = builder.u32_opcode(OPCODE_AND, a_table, b_table);
//         let op = builder.u32_operation_from_opcode(&opcode_table);
//         let op_write = U32OperationWrite::new(opcode_table.clone(), op);

//         // Set up the bus
//         let mut bus = builder.new_bus();
//         let challenges = builder.alloc_challenge_array::<CubicRegister>(4);

//         let mut write_channel = builder.new_u32_channel(&challenges, &mut bus);
//         let mut read_channel = builder.new_u32_channel(&challenges, &mut bus);

//         builder.input_to_u32_channel(&opcode_dst, &mut read_channel);
//         builder.output_from_u32_channel(&opcode_table, &mut write_channel);

//         // constrain the bus
//         builder.constrain_bus(bus);

//         let air = builder.build();

//         let generator = ArithmeticGenerator::<L>::new(&air);
//         let writer = generator.new_writer();
//         let mut rng = thread_rng();
//         for i in 0..L::num_rows() {
//             let a_val = F::from_canonical_u32(rng.gen::<u32>());
//             let b_val = F::from_canonical_u32(rng.gen::<u32>());

//             writer.write(&a, &a_val, L::num_rows() - 1 - i);
//             writer.write(&b, &b_val, L::num_rows() - 1 - i);

//             writer.write(&a_table, &a_val, i);
//             writer.write(&b_table, &b_val, i);

//             writer.write_instruction(&opcode_dst, L::num_rows() - 1 - i);
//             writer.write_instruction(&op_write, i);
//         }
//         let stark = Starky::<_, { L::num_columns() }>::new(air);
//         let config = SC::standard_fast_config(L::num_rows());

//         // Generate proof and verify as a stark
//         timed!(
//             timing,
//             "stark proof",
//             test_starky(&stark, &config, &generator, &[])
//         );

//         // Test the recursive proof.
//         test_recursive_starky(stark, config, generator, &[]);
//         timing.print();
//     }
// }
