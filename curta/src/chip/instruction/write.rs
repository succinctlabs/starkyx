use std::collections::HashSet;

use super::{Instruction, InstructionId};
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;
use crate::trace::writer::TraceWriter;

#[derive(Debug, Clone, Copy)]
pub struct WriteInstruction(pub MemorySlice);

impl<AP: AirParser> AirConstraint<AP> for WriteInstruction {
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: Field> Instruction<F> for WriteInstruction {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![self.0]
    }

    fn inputs(&self) -> HashSet<MemorySlice> {
        HashSet::new()
    }

    fn constraint_degree(&self) -> usize {
        0
    }

    fn write(&self, _writer: &TraceWriter<F>, _row_index: usize) {}

    fn id(&self) -> InstructionId {
        InstructionId::Write(self.0)
    }
}
