use serde::{Deserialize, Serialize};

use self::builder_operations::ByteLookupOperations;
use self::table::ByteLookupTable;
use super::bit_operations::and::And;
use super::bit_operations::not::Not;
use super::bit_operations::xor::Xor;
use super::decode::ByteDecodeInstruction;
use super::operations::instruction::ByteOperationInstruction;
use super::operations::value::ByteOperationDigestConstraint;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::bool::SelectInstruction;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
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
    BitSelect(SelectInstruction<BitRegister>),
    Decode(ByteDecodeInstruction),
    Digest(ByteOperationDigestConstraint),
}

pub trait ByteInstructions:
    From<ByteInstructionSet>
    + From<ByteOperationInstruction>
    + From<SelectInstruction<BitRegister>>
    + From<ByteDecodeInstruction>
    + From<ByteOperationDigestConstraint>
{
}

impl ByteInstructions for ByteInstructionSet {}

impl<L: AirParameters> AirBuilder<L> {
    pub fn byte_operations(&mut self) -> (ByteLookupOperations, ByteLookupTable)
    where
        L::Instruction: From<ByteInstructionSet>
            + From<SelectInstruction<BitRegister>>
            + From<ByteDecodeInstruction>,
    {
        let lookup_table = self.new_byte_lookup_table();
        let operations =
            ByteLookupOperations::new(lookup_table.multiplicity_data.clone());

        (operations, lookup_table)
    }

    pub fn register_byte_lookup(
        &mut self,
        operation_values: ByteLookupOperations,
        table: &ByteLookupTable,
    ) {
        let multiplicities = table.multiplicity_data.multiplicities();
        let lookup_challenge = self.alloc_challenge::<CubicRegister>();

        let lookup_table = self.lookup_table_with_multiplicities(
            &lookup_challenge,
            &table.digests,
            multiplicities,
        );
        let lookup_values = self.lookup_values(&lookup_challenge, &operation_values.values);

        self.element_lookup_from_table_and_values(lookup_table, lookup_values);
    }
}

impl<AP: AirParser> AirConstraint<AP> for ByteInstructionSet {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Op(op) => op.eval(parser),
            Self::BitAnd(op) => op.eval(parser),
            Self::BitXor(op) => op.eval(parser),
            Self::BitNot(op) => op.eval(parser),
            Self::BitSelect(op) => op.eval(parser),
            Self::Decode(instruction) => instruction.eval(parser),
            Self::Digest(instruction) => instruction.eval(parser),
        }
    }
}

impl<F: PrimeField64> Instruction<F> for ByteInstructionSet {
    fn inputs(&self) -> Vec<MemorySlice> {
        match self {
            Self::Op(op) => Instruction::<F>::inputs(op),
            Self::BitAnd(op) => Instruction::<F>::inputs(op),
            Self::BitXor(op) => Instruction::<F>::inputs(op),
            Self::BitNot(op) => Instruction::<F>::inputs(op),
            Self::BitSelect(op) => Instruction::<F>::inputs(op),
            Self::Decode(instruction) => Instruction::<F>::inputs(instruction),
            Self::Digest(instruction) => Instruction::<F>::inputs(instruction),
        }
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            Self::Op(op) => Instruction::<F>::trace_layout(op),
            Self::BitAnd(op) => Instruction::<F>::trace_layout(op),
            Self::BitXor(op) => Instruction::<F>::trace_layout(op),
            Self::BitNot(op) => Instruction::<F>::trace_layout(op),
            Self::BitSelect(op) => Instruction::<F>::trace_layout(op),
            Self::Decode(instruction) => Instruction::<F>::trace_layout(instruction),
            Self::Digest(instruction) => Instruction::<F>::trace_layout(instruction),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Self::Op(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitAnd(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitXor(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitNot(op) => Instruction::<F>::write(op, writer, row_index),
            Self::BitSelect(op) => Instruction::<F>::write(op, writer, row_index),
            Self::Decode(instruction) => Instruction::<F>::write(instruction, writer, row_index),
            Self::Digest(instruction) => Instruction::<F>::write(instruction, writer, row_index),
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

impl From<SelectInstruction<BitRegister>> for ByteInstructionSet {
    fn from(op: SelectInstruction<BitRegister>) -> Self {
        Self::BitSelect(op)
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
    use crate::chip::builder::AirBuilder;
    use crate::chip::register::Register;
    use crate::chip::uint::bytes::operations::value::ByteOperation;
    use crate::chip::uint::bytes::register::ByteRegister;
    use crate::chip::AirParameters;
    use crate::math::field::Field;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ByteOpTest<const N: usize>;

    impl<const N: usize> AirParameters for ByteOpTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ByteInstructionSet;

        const NUM_FREE_COLUMNS: usize = 377;
        const EXTENDED_COLUMNS: usize = 159;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
    }

    #[test]
    fn test_bit_op_lookup() {
        type F = GoldilocksField;
        const NUM_VALS: usize = 10;
        type L = ByteOpTest<NUM_VALS>;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let (mut operations, table) = builder.byte_operations();

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

        builder.register_byte_lookup(operations, &table);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;

        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        table.write_table_entries(&writer);

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
        table.write_multiplicities(&writer);

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        let public_inputs = writer.public.read().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
