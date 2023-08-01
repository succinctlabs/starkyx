//! U32

pub mod arithmetic;
pub mod bit_operations;
pub mod channel;
pub mod opcode;
pub mod operation;
pub mod write_operation;

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::operation::U32Operation;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::register::cubic::CubicRegister;
    use crate::chip::u32::opcode::OPCODE_AND;
    use crate::chip::u32::write_operation::U32OperationWrite;
    use crate::chip::AirParameters;
    use crate::math::prelude::*;

    #[derive(Debug, Clone)]
    pub struct U32OperationTest;

    impl const AirParameters for U32OperationTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = U32Operation;

        const NUM_FREE_COLUMNS: usize = 104;
        const EXTENDED_COLUMNS: usize = 18;

        fn num_rows_bits() -> usize {
            10
        }
    }

    #[test]
    fn test_and_opcode() {
        type F = GoldilocksField;
        type L = U32OperationTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc::<ElementRegister>();
        let b = builder.alloc::<ElementRegister>();
        let opcode_dst = builder.u32_opcode(OPCODE_AND, a, b);

        let a_table = builder.alloc::<ElementRegister>();
        let b_table = builder.alloc::<ElementRegister>();
        let opcode_table = builder.u32_opcode(OPCODE_AND, a_table, b_table);
        let op = builder.u32_operation_from_opcode(&opcode_table);
        let op_write = U32OperationWrite::new(opcode_table.clone(), op);

        // Set up the bus
        let mut bus = builder.new_bus();
        let challenges = builder.alloc_challenge_array::<CubicRegister>(4);

        let mut write_channel = builder.new_u32_channel(&challenges, &mut bus);
        let mut read_channel = builder.new_u32_channel(&challenges, &mut bus);

        builder.input_to_u32_channel(&opcode_dst, &mut read_channel);
        builder.output_from_u32_channel(&opcode_table, &mut write_channel);

        // constrain the bus
        builder.constrain_bus(bus);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        let mut rng = thread_rng();
        for i in 0..L::num_rows() {
            let a_val = F::from_canonical_u32(rng.gen::<u32>());
            let b_val = F::from_canonical_u32(rng.gen::<u32>());

            writer.write(&a, &a_val, L::num_rows() - 1 - i);
            writer.write(&b, &b_val, L::num_rows() - 1 - i);

            writer.write(&a_table, &a_val, i);
            writer.write(&b_table, &b_val, i);

            writer.write_instruction(&opcode_dst, L::num_rows() - 1 - i);
            writer.write_instruction(&op_write, L::num_rows() - 1 - i);
        }

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
