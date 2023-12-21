use serde::{Deserialize, Serialize};

use super::get::GetInstruction;
use super::set::SetInstruction;
use super::time::Time;
use super::watch::WatchInstruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::field::Field;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryInstruction<F> {
    Get(GetInstruction<F>),
    Set(SetInstruction<F>),
    Watch(WatchInstruction),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOutput<F> {
    pub label: String,
    pub index: Option<MemorySliceIndex>,
    pub ts: Time<F>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemorySliceIndex {
    /// The index of the memory slice.
    Index(usize),
    /// The index of the memory slice and the index of the element within the slice.
    IndexElement(ElementRegister),
}

impl<AP: AirParser> AirConstraint<AP> for MemoryInstruction<AP::Field> {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Get(instr) => instr.eval(parser),
            Self::Set(instr) => instr.eval(parser),
            Self::Watch(instr) => instr.eval(parser),
        }
    }
}

impl<F: Field> Instruction<F> for MemoryInstruction<F> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Self::Get(instr) => instr.write(writer, row_index),
            Self::Set(instr) => instr.write(writer, row_index),
            Self::Watch(instr) => instr.write(writer, row_index),
        }
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        match self {
            Self::Get(instr) => instr.write_to_air(writer),
            Self::Set(instr) => instr.write_to_air(writer),
            Self::Watch(instr) => instr.write_to_air(writer),
        }
    }
}
