use core::fmt::Debug;
use core::hash::Hash;
use std::collections::HashSet;

use super::register::memory::MemorySlice;
use crate::math::prelude::*;
use crate::trace::writer::TraceWriter;
pub mod empty;
pub mod node;
pub mod write;

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum InstructionId {
    CustomInstruction(Vec<MemorySlice>),
    Write(MemorySlice),
}

pub trait Instruction<F: Field>: 'static + Send + Sync + Debug + Clone {
    // Returns a vector of memory slices or contiguous memory regions of the row in the trace that
    // instruction relies on. These registers must be filled in by the `TraceWriter`.
    fn trace_layout(&self) -> Vec<MemorySlice>;

    /// Returns a vector of memory slices or contiguous memory regions of the row in the trace that
    /// specifies the inputs to the instruction.
    fn inputs(&self) -> HashSet<MemorySlice>;

    /// Writes the instruction to the trace.
    ///
    /// This method is called after all the inputs returned from `inputs` have been written to the trace.
    fn write(&self, writer: &TraceWriter<F>);

    fn constraint_degree(&self) -> usize {
        2
    }

    fn id(&self) -> InstructionId {
        InstructionId::CustomInstruction(self.trace_layout())
    }
}
