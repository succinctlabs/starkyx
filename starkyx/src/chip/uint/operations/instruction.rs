use serde::{Deserialize, Serialize};

use super::add::ByteArrayAdd;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::uint::bytes::decode::ByteDecodeInstruction;
use crate::chip::uint::bytes::lookup_table::{ByteInstructionSet, ByteInstructions};
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperationDigestConstraint;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UintInstruction {
    Bit(ByteInstructionSet),
    Add(ByteArrayAdd<4>),
}

pub trait UintInstructions:
    ByteInstructions + From<UintInstruction> + From<ByteArrayAdd<4>>
{
}

impl ByteInstructions for UintInstruction {}

impl UintInstructions for UintInstruction {}

impl<AP: AirParser> AirConstraint<AP> for UintInstruction {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Bit(op) => op.eval(parser),
            Self::Add(op) => op.eval(parser),
        }
    }
}

impl<F: PrimeField64> Instruction<F> for UintInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Self::Bit(op) => Instruction::<F>::write(op, writer, row_index),
            Self::Add(op) => Instruction::<F>::write(op, writer, row_index),
        }
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        match self {
            Self::Bit(op) => Instruction::<F>::write_to_air(op, writer),
            Self::Add(op) => Instruction::<F>::write_to_air(op, writer),
        }
    }
}

impl From<ByteInstructionSet> for UintInstruction {
    fn from(op: ByteInstructionSet) -> Self {
        Self::Bit(op)
    }
}

impl From<ByteArrayAdd<4>> for UintInstruction {
    fn from(op: ByteArrayAdd<4>) -> Self {
        Self::Add(op)
    }
}

impl From<ByteOperationInstruction> for UintInstruction {
    fn from(op: ByteOperationInstruction) -> Self {
        Self::Bit(op.into())
    }
}

impl From<ByteDecodeInstruction> for UintInstruction {
    fn from(op: ByteDecodeInstruction) -> Self {
        Self::Bit(op.into())
    }
}

impl From<ByteOperationDigestConstraint> for UintInstruction {
    fn from(op: ByteOperationDigestConstraint) -> Self {
        Self::Bit(op.into())
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::register::bit::BitRegister;
    use crate::chip::uint::register::ByteArrayRegister;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct U32OpTest;

    impl AirParameters for U32OpTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 1100;
        const EXTENDED_COLUMNS: usize = 1500;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct U64OpTest;

    impl AirParameters for U64OpTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 2200;
        const EXTENDED_COLUMNS: usize = 2700;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
    }

    #[test]
    fn test_u32_bit_operations() {
        type F = GoldilocksField;
        const N: usize = 4;
        type L = U32OpTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let mut operations = builder.byte_operations();

        let a = builder.alloc::<ByteArrayRegister<N>>();
        let b = builder.alloc::<ByteArrayRegister<N>>();

        let a_and_b = builder.bitwise_and(&a, &b, &mut operations);
        let and_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_and_b, &and_expected);

        let a_xor_b = builder.bitwise_xor(&a, &b, &mut operations);
        let xor_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_xor_b, &xor_expected);

        let a_not = builder.bitwise_not(&a, &mut operations);
        let not_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_not, &not_expected);

        let (a_plus_b, carry) = builder.carrying_add_u32(&a, &b, &None, &mut operations);
        let add_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_plus_b, &add_expected);
        let carry_expected = builder.alloc::<BitRegister>();
        builder.assert_equal(&carry, &carry_expected);

        let mut rng = thread_rng();

        let mut shr_shift_vals = vec![];
        let mut shr_expected_vec = vec![];
        let mut rot_expected_vec = vec![];

        let num_ops = 20;

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

        let mut byte_table = builder.new_byte_lookup_table();
        let byte_data = builder.register_byte_lookup(&mut byte_table, operations);
        builder.constraint_byte_lookup_table(&byte_table);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;

        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let to_field = |a: u32| a.to_le_bytes().map(F::from_canonical_u8);

        byte_table.write_table_entries(&writer);
        let mut rng = thread_rng();
        for i in 0..num_rows {
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

            let (add_val, carry_val) = a_val.carrying_add(b_val, false);
            writer.write(&add_expected, &to_field(add_val), i);
            writer.write(&carry_expected, &F::from_canonical_u8(carry_val as u8), i);

            for k in 0..num_ops {
                let shr_val = a_val >> shr_shift_vals[k];
                writer.write(&shr_expected_vec[k], &to_field(shr_val), i);
            }

            for k in 0..num_ops {
                let rot_val = a_val.rotate_right(shr_shift_vals[k] as u32);
                writer.write(&rot_expected_vec[k], &to_field(rot_val), i);
            }

            writer.write_row_instructions(&generator.air_data, i);
        }
        let multiplicities = byte_data.get_multiplicities(&writer);
        writer.write_lookup_multiplicities(byte_table.multiplicities(), &[multiplicities]);

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }

    #[test]
    fn test_u64_bit_operations() {
        type F = GoldilocksField;
        const N: usize = 8;
        type L = U64OpTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let mut operations = builder.byte_operations();

        let a = builder.alloc::<ByteArrayRegister<N>>();
        let b = builder.alloc::<ByteArrayRegister<N>>();

        let a_and_b = builder.bitwise_and(&a, &b, &mut operations);
        let and_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_and_b, &and_expected);

        let a_xor_b = builder.bitwise_xor(&a, &b, &mut operations);
        let xor_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_xor_b, &xor_expected);

        let a_not = builder.bitwise_not(&a, &mut operations);
        let not_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_not, &not_expected);

        let (a_plus_b, carry) = builder.carrying_add_u64(&a, &b, &None, &mut operations);
        let add_expected = builder.alloc::<ByteArrayRegister<N>>();
        builder.assert_equal(&a_plus_b, &add_expected);
        let carry_expected = builder.alloc::<BitRegister>();
        builder.assert_equal(&carry, &carry_expected);

        let mut rng = thread_rng();

        let mut shr_shift_vals = vec![];
        let mut shr_expected_vec = vec![];
        let mut rot_expected_vec = vec![];

        let num_ops = 20;

        for i in 0..num_ops {
            let shift = if i == 0 {
                32usize
            } else {
                rng.gen::<u64>() as usize
            };
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

        let mut byte_table = builder.new_byte_lookup_table();
        let byte_data = builder.register_byte_lookup(&mut byte_table, operations);
        builder.constraint_byte_lookup_table(&byte_table);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;

        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let to_field = |a: u64| a.to_le_bytes().map(F::from_canonical_u8);

        byte_table.write_table_entries(&writer);
        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_val = rng.gen::<u64>();
            let b_val = rng.gen::<u64>();
            writer.write(&a, &to_field(a_val), i);
            writer.write(&b, &to_field(b_val), i);

            let and_val = a_val & b_val;
            writer.write(&and_expected, &to_field(and_val), i);

            let xor_val = a_val ^ b_val;
            writer.write(&xor_expected, &to_field(xor_val), i);

            let not_val = !a_val;
            writer.write(&not_expected, &to_field(not_val), i);

            let (add_val, carry_val) = a_val.carrying_add(b_val, false);
            writer.write(&add_expected, &to_field(add_val), i);
            writer.write(&carry_expected, &F::from_canonical_u8(carry_val as u8), i);

            for k in 0..num_ops {
                let shr_val = a_val >> shr_shift_vals[k];
                writer.write(&shr_expected_vec[k], &to_field(shr_val), i);
            }

            for k in 0..num_ops {
                let rot_val = a_val.rotate_right(shr_shift_vals[k] as u32);
                writer.write(&rot_expected_vec[k], &to_field(rot_val), i);
            }

            writer.write_row_instructions(&generator.air_data, i);
        }
        let multiplicities = byte_data.get_multiplicities(&writer);
        writer.write_lookup_multiplicities(byte_table.multiplicities(), &[multiplicities]);

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
