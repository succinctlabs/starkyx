use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::instruction::Instruction;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::lookup_table::builder_operations::ByteLookupOperations;
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::register::{ByteArrayRegister, U32Register};
use crate::chip::AirParameters;
use crate::math::prelude::*;

///
///
/// Assumes 2^N < FIELD_SIZE
#[derive(Debug, Clone)]
pub struct ByteArrayAdd<const N: usize> {
    pub a: ByteArrayRegister<N>,
    pub b: ByteArrayRegister<N>,
    pub result: ByteArrayRegister<N>,
    carry: BitRegister,
}

impl<const N: usize> ByteArrayAdd<N> {
    pub fn new(
        a: ByteArrayRegister<N>,
        b: ByteArrayRegister<N>,
        result: ByteArrayRegister<N>,
        carry: BitRegister,
    ) -> Self {
        Self {
            a,
            b,
            result,
            carry,
        }
    }
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn set_add_u32(
        &mut self,
        a: &U32Register,
        b: &U32Register,
        result: &U32Register,
        operations: &mut ByteLookupOperations,
    ) -> BitRegister
    where
        L::Instruction: From<ByteArrayAdd<4>> + From<ByteOperationInstruction>,
    {
        let carry = self.alloc::<BitRegister>();
        let add = ByteArrayAdd::<4>::new(*a, *b, *result, carry);
        self.register_instruction(add);

        for byte in result.bytes() {
            let result_range = ByteOperation::Range(byte);
            self.set_byte_operation(&result_range, operations);
        }

        carry
    }
}

impl<AP: AirParser, const N: usize> AirConstraint<AP> for ByteArrayAdd<N> {
    fn eval(&self, parser: &mut AP) {
        let a = self.a.eval(parser);
        let b = self.b.eval(parser);
        let result = self.result.eval(parser);
        let carry = self.carry.eval(parser);

        let mut a_val = parser.zero();
        let mut b_val = parser.zero();
        let mut result_val = parser.zero();

        for (i, ((a_byte, b_byte), res_byte)) in a.into_iter().zip(b).zip(result).enumerate() {
            let mult = AP::Field::from_canonical_u32(1 << (8 * i));
            let a_byte_times_mult = parser.mul_const(a_byte, mult);
            let b_byte_times_mult = parser.mul_const(b_byte, mult);
            let res_byte_times_mult = parser.mul_const(res_byte, mult);

            a_val = parser.add(a_val, a_byte_times_mult);
            b_val = parser.add(b_val, b_byte_times_mult);
            result_val = parser.add(result_val, res_byte_times_mult);
        }

        let a_plus_b = parser.add(a_val, b_val);
        let two_power = AP::Field::from_canonical_u64(1 << (8 * N));
        let carry_times_mod = parser.mul_const(carry, two_power);
        let result_plus_carry = parser.add(result_val, carry_times_mod);
        let constraint = parser.sub(a_plus_b, result_plus_carry);
        parser.constraint(constraint);
    }
}

impl<F: PrimeField64> Instruction<F> for ByteArrayAdd<4> {
    fn inputs(&self) -> Vec<MemorySlice> {
        vec![*self.a.register(), *self.b.register()]
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![*self.result.register(), *self.carry.register()]
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let a = writer.read(&self.a, row_index);
        let b = writer.read(&self.b, row_index);

        let a_val = u32::from_le_bytes(a.map(|x| x.as_canonical_u64() as u8));
        let b_val = u32::from_le_bytes(b.map(|x| x.as_canonical_u64() as u8));

        let (result, carry) = a_val.carrying_add(b_val, false);
        let result_bytes = result.to_le_bytes().map(|x| F::from_canonical_u8(x));

        writer.write(&self.result, &result_bytes, row_index);
        writer.write(&self.carry, &F::from_canonical_u8(carry as u8), row_index);
    }
}
