use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::register::ByteArrayRegister;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_bitwise_and<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        b: &ByteArrayRegister<N>,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        for ((a_byte, b_byte), result_byte) in a
            .bytes()
            .iter()
            .zip(b.bytes().iter())
            .zip(result.bytes().iter())
        {
            let and = ByteOperation::And(a_byte, b_byte, result_byte);
            self.set_byte_operation(&and, operations);
        }
    }

    pub fn set_bitwise_xor<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        b: &ByteArrayRegister<N>,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        for ((a_byte, b_byte), result_byte) in a
            .bytes()
            .iter()
            .zip(b.bytes().iter())
            .zip(result.bytes().iter())
        {
            let xor = ByteOperation::Xor(a_byte, b_byte, result_byte);
            self.set_byte_operation(&xor, operations);
        }
    }

    pub fn set_bitwise_not<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        for (a_byte, result_byte) in a.bytes().iter().zip(result.bytes().iter()) {
            let not = ByteOperation::Not(a_byte, result_byte);
            self.set_byte_operation(&not, operations);
        }
    }

    pub fn set_bit_shr<const N: usize>(
        &mut self,
        a: &ByteArrayRegister<N>,
        shift: usize,
        result: &ByteArrayRegister<N>,
        operations: &mut ByteLookupOperations,
    ) where
        L::Instruction: From<ByteOperationInstruction>,
    {
        let a_bytes = a.bytes();
        let result_bytes = result.bytes();

        let shift = shift % (N * 8);

        let byte_shift = shift / 8;
        let zero = ArithmeticExpression::zero();
        for i in N-byte_shift..N {
            self.set_to_expression(&result_bytes.get(i), zero.clone());
        }

        let bit_shift = shift % 8;
        // for (a_byte, result_byte) in a.bytes().iter().zip(result.bytes().iter()) {
        //     let shr = ByteOperation::Shr(a_byte, shift, result_byte);
        //     self.set_byte_operation(&shr, operations);
        // }
    }
}

#[cfg(test)]
mod tests {
    use num::traits::ToBytes;
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::bytes::lookup_table::ByteInstructionSet;
    use crate::chip::uint::bytes::register::ByteRegister;
    use crate::chip::AirParameters;
    use crate::plonky2::field::Field;

    #[derive(Debug, Clone)]
    struct UintBitOpTest<const N: usize>;

    impl<const N: usize> const AirParameters for UintBitOpTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ByteInstructionSet;

        const NUM_FREE_COLUMNS: usize = 133;
        const EXTENDED_COLUMNS: usize = 96;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_u32_bit_operations() {
        type F = GoldilocksField;
        const N: usize = 4;
        type L = UintBitOpTest<N>;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let (mut operations, mut table) = builder.byte_operations();

        let a = builder.alloc::<ByteArrayRegister<N>>();
        let b = builder.alloc::<ByteArrayRegister<N>>();

        let a_and_b = builder.alloc::<ByteArrayRegister<N>>();
        builder.set_bitwise_and(&a, &b, &a_and_b, &mut operations);
        let and_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_and_b, &and_expected); 

        let a_xor_b = builder.alloc::<ByteArrayRegister<N>>();
        builder.set_bitwise_xor(&a, &b, &a_xor_b, &mut operations);
        let xor_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_xor_b, &xor_expected);

        let a_not = builder.alloc::<ByteArrayRegister<N>>();
        builder.set_bitwise_not(&a, &a_not, &mut operations);
        let not_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_not, &not_expected);

        builder.register_byte_lookup(operations, &table);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        table.write_table_entries(&writer);

        let mut rng = thread_rng();

        let to_field = |a : u32| a.to_le_bytes().map(|x| F::from_canonical_u8(x));
        for i in 0..L::num_rows() {
            let a_val = rng.gen::<u32>();
            let b_val = rng.gen::<u32>();
            writer.write(&a, &to_field(a_val), i);
            writer.write(&b, &to_field(b_val), i);

            let and_val = a_val & b_val;
            writer.write(&and_expected, &to_field(and_val), i);

            let xor_val = a_val ^ b_val;
            writer.write(&xor_expected, &to_field(xor_val), i);

            let not_val = !a_val;
            writer.write(&not_expected, &to_field(not_val), i);

            writer.write_row_instructions(&air, i);
        }

        table.write_multiplicities(&writer, L::num_rows() * 2);

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
