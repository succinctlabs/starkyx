use super::Instruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

/// A defult instruction set that contains no custom instructions
#[derive(Clone, Debug)]
pub struct EmptyInstruction<F> {
    _marker: core::marker::PhantomData<F>,
}

impl<F: Field> Instruction<F> for EmptyInstruction<F> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        Vec::new()
    }

    fn inputs(&self) -> Vec<MemorySlice> {
        Vec::new()
    }

    fn write(&self, _writer: &TraceWriter<F>, _row_index: usize) {}
}

impl<F: Field, AP: AirParser<Field = F>> AirConstraint<AP> for EmptyInstruction<F> {
    fn eval(&self, _parser: &mut AP) {}
}
