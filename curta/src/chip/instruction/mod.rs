use core::fmt::Debug;
use core::hash::Hash;
use std::collections::HashSet;

use super::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

pub mod assign;
pub mod bit;
pub mod empty;
pub mod node;
pub mod set;
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
    fn write(&self, writer: &TraceWriter<F>, row_index: usize);

    fn constraint_degree(&self) -> usize {
        2
    }

    fn id(&self) -> InstructionId {
        InstructionId::CustomInstruction(self.trace_layout())
    }
}

/// An instruction that only consists of constraints
pub trait ConstraintInstruction: 'static + Clone + Debug + Send + Sync {}

impl<F: Field, C: ConstraintInstruction> Instruction<F> for C {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![]
    }

    fn inputs(&self) -> HashSet<MemorySlice> {
        HashSet::new()
    }

    fn write(&self, _writer: &TraceWriter<F>, _row_index: usize) {}
}
