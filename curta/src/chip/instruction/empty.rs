use std::collections::HashSet;

use super::Instruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;

/// A defult instruction set that contains no custom instructions
#[derive(Clone, Debug)]
pub struct EmptyInstructionSet<F> {
    _marker: core::marker::PhantomData<F>,
}

impl<F: Field> Instruction<F> for EmptyInstructionSet<F> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        Vec::new()
    }

    fn inputs(&self) -> HashSet<MemorySlice> {
        HashSet::new()
    }

    fn write(&self, _writer: &crate::trace::writer::TraceWriter<F>) {}
}

impl<F: Field, AP: AirParser<Field = F>> AirConstraint<AP> for EmptyInstructionSet<F> {
    fn eval(&self, _parser: &mut AP) {}
}
