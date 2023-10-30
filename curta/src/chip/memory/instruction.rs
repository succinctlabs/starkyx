use serde::{Deserialize, Serialize};

use super::get::GetInstruction;
use super::set::SetInstruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::TraceWriter;
use crate::math::field::Field;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryInstruction {
    Get(GetInstruction),
    Set(SetInstruction),
}

impl<AP: AirParser> AirConstraint<AP> for MemoryInstruction {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Get(instr) => instr.eval(parser),
            Self::Set(instr) => instr.eval(parser),
        }
    }
}

impl<F: Field> Instruction<F> for MemoryInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            Self::Get(instr) => instr.write(writer, row_index),
            Self::Set(instr) => instr.write(writer, row_index),
        }
    }
}
