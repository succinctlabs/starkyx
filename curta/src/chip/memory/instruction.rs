use serde::{Deserialize, Serialize};

use super::get::GetInstruction;
use super::set::SetInstruction;
use super::watch::WatchInstruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::field::Field;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryInstruction {
    Get(GetInstruction),
    Set(SetInstruction),
    Watch(WatchInstruction),
}

impl<AP: AirParser> AirConstraint<AP> for MemoryInstruction {
    fn eval(&self, parser: &mut AP) {
        match self {
            Self::Get(instr) => instr.eval(parser),
            Self::Set(instr) => instr.eval(parser),
            Self::Watch(instr) => instr.eval(parser),
        }
    }
}

impl<F: Field> Instruction<F> for MemoryInstruction {
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
