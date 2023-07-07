use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use std::collections::HashSet;

use dep_graph::{DepGraph, Node};

use super::{Instruction, InstructionId};
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;

pub type InstructionNode<F, I> = Node<WrappedInstruction<F, I>>;

pub type InstructionGraph<F, I> = DepGraph<WrappedInstruction<F, I>>;

#[derive(Debug, Clone)]
pub struct WrappedInstruction<F, I>(pub I, PhantomData<F>);

impl<F: Field, I: Instruction<F>> From<I> for WrappedInstruction<F, I> {
    fn from(instruction: I) -> Self {
        Self(instruction, PhantomData)
    }
}

impl<F: Field, I: Instruction<F>> PartialEq for WrappedInstruction<F, I> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: Field, I: Instruction<F>> Eq for WrappedInstruction<F, I> {}

impl<F: Field, I: Instruction<F>> WrappedInstruction<F, I> {
    pub const fn instruction(&self) -> &I {
        &self.0
    }

    pub fn id(&self) -> InstructionId {
        self.0.id()
    }

    pub fn trace_layout(&self) -> Vec<MemorySlice> {
        self.0.trace_layout()
    }

    pub fn inputs(&self) -> HashSet<MemorySlice> {
        self.0.inputs()
    }
}

impl<F: Field, I: Instruction<F>> Hash for WrappedInstruction<F, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state);
    }
}
