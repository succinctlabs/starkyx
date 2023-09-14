use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::bool::SelectInstruction;
use crate::chip::instruction::Instruction;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::decode::ByteDecodeInstruction;
use crate::chip::uint::bytes::lookup_table::{ByteInstructionSet, ByteInstructions};
use crate::chip::uint::bytes::operations::instruction::ByteOperationInstruction;
use crate::chip::uint::operations::add::ByteArrayAdd;
use crate::chip::uint::operations::instruction::U32Instruction;
use crate::chip::uint::register::U64Register;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum Blake2bInstruction {
    Bit(ByteInstructionSet),
    Add(ByteArrayAdd<4>),
    Select(SelectInstruction<U64Register>),
}

pub trait Blake2bInstructions:
    ByteInstructions
    + From<U32Instruction>
    + From<ByteArrayAdd<4>>
    + From<SelectInstruction<U64Register>>
{
}

impl ByteInstructions for Blake2bInstruction {}

impl<AP: AirParser> AirConstraint<AP> for Blake2bInstruction {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Bit(op) => op.eval(parser),
            Self::Add(op) => op.eval(parser),
            Self::Select(op) => op.eval(parser),
        }
    }
}

impl<F: PrimeField64> Instruction<F> for Blake2bInstruction {
    fn inputs(&self) -> Vec<MemorySlice> {
        match self {
            Self::Bit(op) => Instruction::<F>::inputs(op),
            Self::Add(op) => Instruction::<F>::inputs(op),
            Self::Select(op) => Instruction::<F>::inputs(op),
        }
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            Self::Bit(op) => Instruction::<F>::trace_layout(op),
            Self::Add(op) => Instruction::<F>::trace_layout(op),
            Self::Select(op) => Instruction::<F>::trace_layout(op),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Self::Bit(op) => Instruction::<F>::write(op, writer, row_index),
            Self::Add(op) => Instruction::<F>::write(op, writer, row_index),
            Self::Select(op) => Instruction::<F>::write(op, writer, row_index),
        }
    }
}

impl From<ByteInstructionSet> for Blake2bInstruction {
    fn from(op: ByteInstructionSet) -> Self {
        Self::Bit(op)
    }
}

impl From<ByteArrayAdd<4>> for Blake2bInstruction {
    fn from(op: ByteArrayAdd<4>) -> Self {
        Self::Add(op)
    }
}

impl From<ByteOperationInstruction> for Blake2bInstruction {
    fn from(op: ByteOperationInstruction) -> Self {
        Self::Bit(op.into())
    }
}

impl From<SelectInstruction<BitRegister>> for Blake2bInstruction {
    fn from(op: SelectInstruction<BitRegister>) -> Self {
        Self::Bit(op.into())
    }
}

impl From<ByteDecodeInstruction> for Blake2bInstruction {
    fn from(op: ByteDecodeInstruction) -> Self {
        Self::Bit(op.into())
    }
}
