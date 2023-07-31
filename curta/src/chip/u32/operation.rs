use super::arithmetic::U32ArithmericOperation;
use super::bit_operations::BitOperation;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
pub use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum U32Operation {
    Bit(BitOperation<32>),
    Arithmetic(U32ArithmericOperation),
}

impl<AP: AirParser> AirConstraint<AP> for U32Operation {
    fn eval(&self, parser: &mut AP) {
        match self {
            U32Operation::Bit(op) => op.eval(parser),
            U32Operation::Arithmetic(op) => op.eval(parser),
        }
    }
}

impl<F: PrimeField64> Instruction<F> for U32Operation {
    fn inputs(&self) -> Vec<MemorySlice> {
        match self {
            U32Operation::Bit(op) => Instruction::<F>::inputs(op),
            U32Operation::Arithmetic(op) => Instruction::<F>::inputs(op),
        }
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            U32Operation::Bit(op) => Instruction::<F>::trace_layout(op),
            U32Operation::Arithmetic(op) => Instruction::<F>::trace_layout(op),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            U32Operation::Bit(op) => Instruction::<F>::write(op, writer, row_index),
            U32Operation::Arithmetic(op) => Instruction::<F>::write(op, writer, row_index),
        }
    }
}

impl From<BitOperation<32>> for U32Operation {
    fn from(op: BitOperation<32>) -> Self {
        U32Operation::Bit(op)
    }
}

impl From<U32ArithmericOperation> for U32Operation {
    fn from(op: U32ArithmericOperation) -> Self {
        U32Operation::Arithmetic(op)
    }
}
