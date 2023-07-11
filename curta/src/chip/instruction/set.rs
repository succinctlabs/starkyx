use core::hash::{Hash, Hasher};
use std::collections::HashSet;

use super::assign::AssignInstruction;
use super::bit::BitConstraint;
use super::write::WriteInstruction;
use super::{Instruction, InstructionId};
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum InstructionSet<F, I> {
    CustomInstruction(I),
    WriteInstruction(WriteInstruction),
    BitConstraint(BitConstraint),
    Assign(AssignInstruction<F>),
}

impl<F: Field, AP: AirParser<Field = F>, I: AirConstraint<AP>> AirConstraint<AP>
    for InstructionSet<F, I>
{
    fn eval(&self, parser: &mut AP) {
        match self {
            InstructionSet::CustomInstruction(i) => AirConstraint::<AP>::eval(i, parser),
            InstructionSet::WriteInstruction(i) => AirConstraint::<AP>::eval(i, parser),
            InstructionSet::BitConstraint(i) => AirConstraint::<AP>::eval(i, parser),
            InstructionSet::Assign(i) => AirConstraint::<AP>::eval(i, parser),
        }
    }
}

impl<F: Field, I: Instruction<F>> Instruction<F> for InstructionSet<F, I> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            InstructionSet::CustomInstruction(i) => i.trace_layout(),
            InstructionSet::WriteInstruction(i) => Instruction::<F>::trace_layout(i),
            InstructionSet::BitConstraint(i) => Instruction::<F>::trace_layout(i),
            InstructionSet::Assign(i) => i.trace_layout(),
        }
    }

    fn inputs(&self) -> HashSet<MemorySlice> {
        match self {
            InstructionSet::CustomInstruction(i) => i.inputs(),
            InstructionSet::WriteInstruction(i) => Instruction::<F>::inputs(i),
            InstructionSet::BitConstraint(i) => Instruction::<F>::inputs(i),
            InstructionSet::Assign(i) => Instruction::<F>::inputs(i),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            InstructionSet::CustomInstruction(i) => i.write(writer, row_index),
            InstructionSet::WriteInstruction(i) => Instruction::<F>::write(i, writer, row_index),
            InstructionSet::BitConstraint(i) => Instruction::<F>::write(i, writer, row_index),
            InstructionSet::Assign(i) => Instruction::<F>::write(i, writer, row_index),
        }
    }

    fn constraint_degree(&self) -> usize {
        match self {
            InstructionSet::CustomInstruction(i) => i.constraint_degree(),
            InstructionSet::WriteInstruction(i) => Instruction::<F>::constraint_degree(i),
            InstructionSet::BitConstraint(i) => Instruction::<F>::constraint_degree(i),
            InstructionSet::Assign(i) => Instruction::<F>::constraint_degree(i),
        }
    }

    fn id(&self) -> InstructionId {
        match self {
            InstructionSet::CustomInstruction(i) => i.id(),
            InstructionSet::WriteInstruction(i) => Instruction::<F>::id(i),
            InstructionSet::BitConstraint(i) => Instruction::<F>::id(i),
            InstructionSet::Assign(i) => Instruction::<F>::id(i),
        }
    }
}

impl<F, I> From<I> for InstructionSet<F, I> {
    fn from(instruction: I) -> Self {
        InstructionSet::CustomInstruction(instruction)
    }
}

impl<F, I> InstructionSet<F, I> {
    pub fn write(register: &MemorySlice) -> Self {
        InstructionSet::WriteInstruction(WriteInstruction(*register))
    }

    pub fn bits(register: &MemorySlice) -> Self {
        InstructionSet::BitConstraint(BitConstraint(*register))
    }

    pub fn assign(assignment: AssignInstruction<F>) -> Self {
        InstructionSet::Assign(assignment)
    }
}

impl<F: Field, I: Instruction<F>> PartialEq for InstructionSet<F, I> {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<F: Field, I: Instruction<F>> PartialOrd for InstructionSet<F, I> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id().partial_cmp(&other.id())
    }
}

impl<F: Field, I: Instruction<F>> Hash for InstructionSet<F, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
