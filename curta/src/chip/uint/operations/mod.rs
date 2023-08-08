pub mod and;
pub mod not;
pub mod rotate;
pub mod shr;
pub mod xor;

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::bytes::lookup_table::ByteInstructionSet;
    use crate::chip::uint::register::ByteArrayRegister;
    use crate::chip::AirParameters;
    use crate::plonky2::field::Field;

    #[derive(Debug, Clone)]
    struct UintBitOpTest<const N: usize>;

    impl<const N: usize> const AirParameters for UintBitOpTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ByteInstructionSet;

        const NUM_FREE_COLUMNS: usize = 661;
        const EXTENDED_COLUMNS: usize = 699;
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

        let mut rng = thread_rng();

        let mut shr_shift_vals = vec![];
        let mut shr_expected_vec = vec![];
        let mut rot_expected_vec = vec![];

        let num_ops = 10;

        for _ in 0..num_ops {
            let shift = rng.gen::<u32>() as usize;
            shr_shift_vals.push(shift);

            let a_shr = builder.alloc::<ByteArrayRegister<N>>();
            builder.set_bit_shr(&a, shift, &a_shr, &mut operations);
            let shr_expected = builder.alloc::<ByteArrayRegister<N>>();
            builder.assert_equal(&a_shr, &shr_expected);
            let a_shr_second = builder.alloc::<ByteArrayRegister<N>>(); // To guarantee even number of operations
            builder.set_bit_shr(&a, shift, &a_shr_second, &mut operations);
            shr_expected_vec.push(shr_expected);

            let a_rot = builder.alloc::<ByteArrayRegister<N>>();
            builder.set_bit_rotate_right(&a, shift, &a_rot, &mut operations);
            let rot_expected = builder.alloc::<ByteArrayRegister<N>>();
            builder.assert_equal(&a_rot, &rot_expected);
            rot_expected_vec.push(rot_expected);

            let a_rot_second = builder.alloc::<ByteArrayRegister<N>>(); // To guarantee even number of operations
            builder.set_bit_rotate_right(&a, shift, &a_rot_second, &mut operations);
        }

        builder.register_byte_lookup(operations, &table);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        table.write_table_entries(&writer);

        let to_field = |a: u32| a.to_le_bytes().map(|x| F::from_canonical_u8(x));
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

            for k in 0..num_ops {
                let shr_val = a_val >> shr_shift_vals[k];
                writer.write(&shr_expected_vec[k], &to_field(shr_val), i);
            }

            for k in 0..num_ops {
                let rot_val = a_val.rotate_right(shr_shift_vals[k] as u32);
                writer.write(&rot_expected_vec[k], &to_field(rot_val), i);
            }

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
