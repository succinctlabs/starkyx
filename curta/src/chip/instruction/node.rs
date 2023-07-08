use core::hash::{Hash, Hasher};
use std::collections::HashSet;

use dep_graph::{DepGraph, Node};

use super::set::InstructionSet;
use super::{Instruction, InstructionId};
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;

pub type InstructionNode<F, I> = Node<WrappedInstruction<F, I>>;

pub type InstructionGraph<F, I> = DepGraph<WrappedInstruction<F, I>>;

#[derive(Debug, Clone)]
pub struct WrappedInstruction<F, I> {
    instruction: InstructionSet<F, I>,
    row_index: usize,
}

impl<F: Field, I: Instruction<F>> WrappedInstruction<F, I> {
    pub fn new(instruction: InstructionSet<F, I>, row_index: usize) -> Self {
        Self {
            instruction,
            row_index,
        }
    }
}

impl<F: Field, I: Instruction<F>> From<(I, usize)> for WrappedInstruction<F, I> {
    fn from((instruction, row_index): (I, usize)) -> Self {
        Self::new(InstructionSet::from(instruction), row_index)
    }
}

impl<F: Field, I: Instruction<F>> From<(InstructionSet<F, I>, usize)> for WrappedInstruction<F, I> {
    fn from((instruction, row_index): (InstructionSet<F, I>, usize)) -> Self {
        Self::new(instruction, row_index)
    }
}

impl<F: Field, I: Instruction<F>> PartialEq for WrappedInstruction<F, I> {
    fn eq(&self, other: &Self) -> bool {
        self.instruction.id() == other.instruction.id() && self.row_index == other.row_index
    }
}

impl<F: Field, I: Instruction<F>> Eq for WrappedInstruction<F, I> {}

impl<F: Field, I: Instruction<F>> WrappedInstruction<F, I> {
    pub const fn instruction(&self) -> &InstructionSet<F, I> {
        &self.instruction
    }

    pub fn id(&self) -> InstructionId {
        self.instruction.id()
    }

    pub fn trace_layout(&self) -> Vec<MemorySlice> {
        self.instruction.trace_layout()
    }

    pub fn inputs(&self) -> HashSet<MemorySlice> {
        self.instruction.inputs()
    }
}

impl<F: Field, I: Instruction<F>> Hash for WrappedInstruction<F, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.row_index.hash(state);
        self.instruction.id().hash(state);
    }
}
