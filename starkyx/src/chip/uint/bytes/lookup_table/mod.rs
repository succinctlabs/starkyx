use serde::{Deserialize, Serialize};

use self::builder_operations::ByteLookupOperations;
use self::multiplicity_data::ByteMultiplicityData;
use self::table::ByteLogLookupTable;
use super::bit_operations::and::And;
use super::bit_operations::not::Not;
use super::bit_operations::xor::Xor;
use super::decode::ByteDecodeInstruction;
use super::operations::instruction::ByteOperationInstruction;
use super::operations::value::ByteOperationDigestConstraint;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::AirParameters;

pub mod builder_operations;
pub mod multiplicity_data;
pub mod table;

use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ByteInstructionSet {
    Op(ByteOperationInstruction),
    BitAnd(And<8>),
    BitXor(Xor<8>),
    BitNot(Not<8>),
    Decode(ByteDecodeInstruction),
    Digest(ByteOperationDigestConstraint),
}

pub trait ByteInstructions:
    From<ByteInstructionSet>
    + From<ByteOperationInstruction>
    + From<ByteDecodeInstruction>
    + From<ByteOperationDigestConstraint>
{
}

impl ByteInstructions for ByteInstructionSet {}

impl<L: AirParameters> AirBuilder<L> {
    pub fn byte_operations(&mut self) -> ByteLookupOperations
    where
        L::Instruction: From<ByteInstructionSet> + From<ByteDecodeInstruction>,
    {
        ByteLookupOperations::new()
    }

    pub fn register_byte_lookup(
        &mut self,
        table: &mut ByteLogLookupTable<L::Field, L::CubicParams>,
        operations: ByteLookupOperations,
    ) -> ByteMultiplicityData {
        let trace_digest_values = operations
            .trace_operations
            .iter()
            .map(|op| self.accumulate_expressions(&table.challenges, &op.expressions()))
            .collect::<Vec<_>>();

        let public_digest_values = operations
            .public_operations
            .iter()
            .map(|op| self.accumulate_public_expressions(&table.challenges, &op.expressions()))
            .collect::<Vec<_>>();

        let values = [trace_digest_values, public_digest_values].concat();

        let _ = table.lookup.register_lookup_values(self, &values);

        ByteMultiplicityData::new(
            table.multiplicity_data.clone(),
            operations.trace_operations.clone(),
            operations.public_operations.clone(),
        )
    }

    pub fn constraint_byte_lookup_table(
        &mut self,
        table: &ByteLogLookupTable<L::Field, L::CubicParams>,
    ) {
        self.constrain_cubic_lookup_table(table.lookup.clone())
    }
}

impl<AP: AirParser> AirConstraint<AP> for ByteInstructionSet {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Op(op) => op.eval(parser),
            Self::BitAnd(op) => op.eval(parser),
            Self::BitXor(op) => op.eval(parser),
            Self::BitNot(op) => op.eval(parser),
            Self::Decode(instruction) => instruction.eval(parser),
            Self::Digest(instruction) => instruction.eval(parser),
        }
    }
}

impl<F: PrimeField64> Instruction<F> for ByteInstructionSet {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Self::Op(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitAnd(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitXor(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitNot(op) => Instruction::<F>::write(op, writer, row_index),
            Self::Decode(instruction) => Instruction::<F>::write(instruction, writer, row_index),
            Self::Digest(instruction) => Instruction::<F>::write(instruction, writer, row_index),
        }
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        match self {
            Self::Op(op) => Instruction::<F>::write_to_air(op, writer),
            Self::BitAnd(op) => Instruction::<F>::write_to_air(op, writer),
            Self::BitXor(op) => Instruction::<F>::write_to_air(op, writer),
            Self::BitNot(op) => Instruction::<F>::write_to_air(op, writer),
            Self::Decode(instruction) => Instruction::<F>::write_to_air(instruction, writer),
            Self::Digest(instruction) => Instruction::<F>::write_to_air(instruction, writer),
        }
    }
}

impl From<ByteOperationInstruction> for ByteInstructionSet {
    fn from(op: ByteOperationInstruction) -> Self {
        Self::Op(op)
    }
}

impl From<And<8>> for ByteInstructionSet {
    fn from(op: And<8>) -> Self {
        Self::BitAnd(op)
    }
}

impl From<Xor<8>> for ByteInstructionSet {
    fn from(op: Xor<8>) -> Self {
        Self::BitXor(op)
    }
}

impl From<Not<8>> for ByteInstructionSet {
    fn from(op: Not<8>) -> Self {
        Self::BitNot(op)
    }
}

impl From<ByteDecodeInstruction> for ByteInstructionSet {
    fn from(instruction: ByteDecodeInstruction) -> Self {
        Self::Decode(instruction)
    }
}

impl From<ByteOperationDigestConstraint> for ByteInstructionSet {
    fn from(instruction: ByteOperationDigestConstraint) -> Self {
        Self::Digest(instruction)
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::register::Register;
    use crate::chip::uint::bytes::operations::value::ByteOperation;
    use crate::chip::uint::bytes::register::ByteRegister;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ByteOpTest<const N: usize>;

    impl<const N: usize> AirParameters for ByteOpTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ByteInstructionSet;

        const NUM_FREE_COLUMNS: usize = 195;
        const EXTENDED_COLUMNS: usize = 453;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
    }

    #[test]
    fn test_bit_op_lookup() {
        type F = GoldilocksField;
        const NUM_VALS: usize = 10;
        type L = ByteOpTest<NUM_VALS>;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let mut byte_table = builder.new_byte_lookup_table();
        let mut operations = builder.byte_operations();

        let mut a_vec = Vec::new();
        let mut b_vec = Vec::new();
        let mut b_const_vec = Vec::new();
        let mut and_expected_vec = Vec::new();
        let mut xor_expected_vec = Vec::new();
        let mut not_expected_vec = Vec::new();
        let mut shr_expected_vec = Vec::new();
        let mut shr_const_expected_vec = Vec::new();
        let mut rot_expected_vec = Vec::new();
        let mut rot_const_expected_vec = Vec::new();

        let mut rng = thread_rng();
        for _ in 0..NUM_VALS {
            let a = builder.alloc::<ByteRegister>();
            let b = builder.alloc::<ByteRegister>();
            a_vec.push(a);
            b_vec.push(b);

            let b_const = rng.gen::<u8>() & 0x7;
            b_const_vec.push(b_const);

            let a_and_b = builder.alloc::<ByteRegister>();
            let and = ByteOperation::And(a, b, a_and_b);
            builder.set_byte_operation(&and, &mut operations);
            let and_expected = builder.alloc::<ByteRegister>();
            builder.assert_equal(&a_and_b, &and_expected);
            and_expected_vec.push(and_expected);

            let a_xor_b = builder.alloc::<ByteRegister>();
            let xor = ByteOperation::Xor(a, b, a_xor_b);
            builder.set_byte_operation(&xor, &mut operations);
            let xor_expected = builder.alloc::<ByteRegister>();
            builder.assert_equal(&a_xor_b, &xor_expected);
            xor_expected_vec.push(xor_expected);

            let a_not = builder.alloc::<ByteRegister>();
            let not = ByteOperation::Not(a, a_not);
            builder.set_byte_operation(&not, &mut operations);
            let not_expected = builder.alloc::<ByteRegister>();
            builder.assert_equal(&a_not, &not_expected);
            not_expected_vec.push(not_expected);

            let a_shr_b = builder.alloc::<ByteRegister>();
            let shr = ByteOperation::Shr(a, b, a_shr_b);
            builder.set_byte_operation(&shr, &mut operations);
            let shr_expected = builder.alloc::<ByteRegister>();
            builder.assert_equal(&a_shr_b, &shr_expected);
            shr_expected_vec.push(shr_expected);

            let a_shr_b_const = builder.alloc::<ByteRegister>();
            let shr = ByteOperation::ShrConst(a, b_const, a_shr_b_const);
            builder.set_byte_operation(&shr, &mut operations);
            let shr_const_expected = builder.alloc::<ByteRegister>();
            builder.assert_equal(&a_shr_b_const, &shr_const_expected);
            shr_const_expected_vec.push(shr_const_expected);

            let a_rot_b = builder.alloc::<ByteRegister>();
            let rot = ByteOperation::Rot(a, b, a_rot_b);
            builder.set_byte_operation(&rot, &mut operations);
            let rot_expected = builder.alloc::<ByteRegister>();
            builder.assert_equal(&a_rot_b, &rot_expected);
            rot_expected_vec.push(rot_expected);

            let a_shr_b_res = builder.alloc::<ByteRegister>();
            let a_shr_b_carry = builder.alloc::<ByteRegister>();
            let shr_carry = ByteOperation::ShrCarry(a, b_const, a_shr_b_res, a_shr_b_carry);
            builder.set_byte_operation(&shr_carry, &mut operations);
            builder.assert_equal(&a_shr_b_res, &shr_const_expected);

            let a_rot_b_const = builder.alloc::<ByteRegister>();
            let rot = ByteOperation::RotConst(a, b_const, a_rot_b_const);
            builder.set_byte_operation(&rot, &mut operations);
            let rot_const_expected = builder.alloc::<ByteRegister>();
            builder.assert_equal(&a_rot_b_const, &rot_const_expected);
            rot_const_expected_vec.push(rot_const_expected);

            let range_op = ByteOperation::Range(a);
            builder.set_byte_operation(&range_op, &mut operations);
        }

        // Add some public byte operations
        let a_pub = builder.alloc_public::<ByteRegister>();
        let b_pub = builder.alloc_public::<ByteRegister>();
        let a_pub_and_b_pub = builder.alloc_public::<ByteRegister>();
        let and_pub = ByteOperation::And(a_pub, b_pub, a_pub_and_b_pub);
        builder.set_public_inputs_byte_operation(&and_pub, &mut operations);
        let a_pub_xor_b_pub = builder.alloc_public::<ByteRegister>();
        let xor_pub = ByteOperation::Xor(a_pub, b_pub, a_pub_xor_b_pub);
        builder.set_public_inputs_byte_operation(&xor_pub, &mut operations);
        let a_not = builder.alloc_public::<ByteRegister>();
        let not = ByteOperation::Not(a_pub, a_not);
        builder.set_public_inputs_byte_operation(&not, &mut operations);
        let b_not = builder.alloc_public::<ByteRegister>();
        let not = ByteOperation::Not(b_pub, b_not);
        builder.set_public_inputs_byte_operation(&not, &mut operations);

        let byte_mult_data = builder.register_byte_lookup(&mut byte_table, operations);
        builder.constraint_byte_lookup_table(&byte_table);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;

        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        // Write public inputs
        let mut public_write = writer.public.write().unwrap();
        let a_pub_val = rng.gen::<u8>();
        let b_pub_val = rng.gen::<u8>();
        let a_pub_and_b_pub_val = a_pub_val & b_pub_val;
        let a_pub_xor_b_pub_val = a_pub_val ^ b_pub_val;
        a_pub.assign_to_raw_slice(&mut public_write, &F::from_canonical_u8(a_pub_val));
        b_pub.assign_to_raw_slice(&mut public_write, &F::from_canonical_u8(b_pub_val));
        a_pub_and_b_pub.assign_to_raw_slice(
            &mut public_write,
            &F::from_canonical_u8(a_pub_and_b_pub_val),
        );
        a_pub_xor_b_pub.assign_to_raw_slice(
            &mut public_write,
            &F::from_canonical_u8(a_pub_xor_b_pub_val),
        );
        a_not.assign_to_raw_slice(&mut public_write, &F::from_canonical_u8(!a_pub_val));
        b_not.assign_to_raw_slice(&mut public_write, &F::from_canonical_u8(!b_pub_val));
        drop(public_write);

        byte_table.write_table_entries(&writer);
        for i in 0..num_rows {
            let mut rng = thread_rng();
            for k in 0..NUM_VALS {
                let a_v = rng.gen::<u8>();
                let b_v = rng.gen::<u8>();
                writer.write(&a_vec[k], &F::from_canonical_u8(a_v), i);
                writer.write(&b_vec[k], &F::from_canonical_u8(b_v), i);

                writer.write(&and_expected_vec[k], &F::from_canonical_u8(a_v & b_v), i);
                writer.write(&xor_expected_vec[k], &F::from_canonical_u8(a_v ^ b_v), i);
                writer.write(&not_expected_vec[k], &F::from_canonical_u8(!a_v), i);
                writer.write(
                    &shr_expected_vec[k],
                    &F::from_canonical_u8(a_v >> (b_v & 0x7)),
                    i,
                );
                writer.write(
                    &shr_const_expected_vec[k],
                    &F::from_canonical_u8(a_v >> b_const_vec[k]),
                    i,
                );
                writer.write(
                    &rot_expected_vec[k],
                    &F::from_canonical_u8(a_v.rotate_right((b_v & 0x7) as u32)),
                    i,
                );
                writer.write(
                    &rot_const_expected_vec[k],
                    &F::from_canonical_u8(a_v.rotate_right(b_const_vec[k] as u32)),
                    i,
                );
            }
            writer.write_row_instructions(&generator.air_data, i);
        }
        writer.write_global_instructions(&generator.air_data);
        let multiplicities = byte_mult_data.get_multiplicities(&writer);
        writer.write_lookup_multiplicities(byte_table.multiplicities(), &[multiplicities]);

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);
        let public_inputs = writer.public.read().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
