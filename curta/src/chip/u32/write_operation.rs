use itertools::Itertools;

use super::opcode::U32Opcode;
use super::operation::U32Operation;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct U32OperationWrite {
    pub opcode: U32Opcode,
    pub operation: U32Operation,
}

impl U32OperationWrite {
    pub fn new(opcode: U32Opcode, operation: U32Operation) -> Self {
        Self { opcode, operation }
    }
}

impl<AP: AirParser> AirConstraint<AP> for U32OperationWrite {
    fn eval(&self, parser: &mut AP) {
        // First, constrain the operation itself
        self.operation.eval(parser);

        // Constrain the compatibility between the opocode entries and the operation bits
        let a_bits: [_; 32] = self.operation.a_bits().eval_array(parser);
        let b_bits: [_; 32] = self.operation.b_bits().eval_array(parser);
        let result_bits: [_; 32] = self.operation.result_bits().eval_array(parser);

        let a = self.opcode.a.eval(parser);
        let b = self.opcode.b.eval(parser);
        let result = self.opcode.result.eval(parser);

        let mut a_acc = parser.zero();
        let mut b_acc = parser.zero();
        let mut result_acc = parser.zero();

        for (i, ((a, b), result)) in a_bits
            .into_iter()
            .zip_eq(b_bits)
            .zip_eq(result_bits)
            .enumerate()
        {
            let two_i = AP::Field::from_canonical_u32(1 << i);
            let a_two_i = parser.mul_const(a, two_i);
            let b_two_i = parser.mul_const(b, two_i);
            let result_two_i = parser.mul_const(result, two_i);

            a_acc = parser.add(a_acc, a_two_i);
            b_acc = parser.add(b_acc, b_two_i);
            result_acc = parser.add(result_acc, result_two_i);
        }

        parser.assert_eq(a, a_acc);
        parser.assert_eq(b, b_acc);
        parser.assert_eq(result, result_acc);
    }
}

impl<F: PrimeField64> Instruction<F> for U32OperationWrite {
    fn inputs(&self) -> Vec<MemorySlice> {
        Instruction::<F>::inputs(&self.opcode)
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        Instruction::<F>::trace_layout(&self.opcode)
            .into_iter()
            .chain(Instruction::<F>::trace_layout(&self.operation))
            .collect()
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        // Write opcode
        Instruction::<F>::write(&self.opcode, writer, row_index);

        // Decompose the operation arguments into bits and write them to the trace
        let a = writer.read(&self.opcode.a, row_index).as_canonical_u64() as u32;
        let b = writer.read(&self.opcode.b, row_index).as_canonical_u64() as u32;
        let result = writer
            .read(&self.opcode.result, row_index)
            .as_canonical_u64() as u32;

        for (i, ((a_bit, b_bit), res_bit)) in self
            .operation
            .a_bits()
            .into_iter()
            .zip_eq(self.operation.b_bits())
            .zip_eq(self.operation.result_bits())
            .enumerate()
        {
            let a_val = F::from_canonical_u32((a >> i) & 1);
            let b_val = F::from_canonical_u32((b >> i) & 1);
            let res_val = F::from_canonical_u32((result >> i) & 1);

            writer.write(&a_bit, &a_val, row_index);
            writer.write(&b_bit, &b_val, row_index);
            writer.write(&res_bit, &res_val, row_index);
        }

        // Write the underlying operation witness (if needed)
        Instruction::<F>::write(&self.operation, writer, row_index);
    }
}
