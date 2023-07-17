use alloc::collections::BTreeSet;
use core::hash::{Hash, Hasher};
use std::collections::HashSet;

use super::assign::AssignInstruction;
use super::bit::BitConstraint;
use super::cycle::Cycle;
use super::write::WriteInstruction;
use super::{Instruction, InstructionId};
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum AirInstruction<F, I> {
    CustomInstruction(I),
    WriteInstruction(WriteInstruction),
    BitConstraint(BitConstraint),
    Assign(AssignInstruction<F>),
    Cycle(Cycle<F>),
}

pub type InstructionSet<L> =
    BTreeSet<AirInstruction<<L as AirParameters>::Field, <L as AirParameters>::Instruction>>;

impl<F: Field, AP: AirParser<Field = F>, I: AirConstraint<AP>> AirConstraint<AP>
    for AirInstruction<F, I>
{
    fn eval(&self, parser: &mut AP) {
        match self {
            AirInstruction::CustomInstruction(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::WriteInstruction(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::BitConstraint(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Assign(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Cycle(i) => AirConstraint::<AP>::eval(i, parser),
        }
    }
}

impl<F: Field, I: Instruction<F>> Instruction<F> for AirInstruction<F, I> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        match self {
            AirInstruction::CustomInstruction(i) => i.trace_layout(),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::trace_layout(i),
            AirInstruction::BitConstraint(i) => Instruction::<F>::trace_layout(i),
            AirInstruction::Assign(i) => i.trace_layout(),
            AirInstruction::Cycle(i) => i.trace_layout(),
        }
    }

    fn inputs(&self) -> HashSet<MemorySlice> {
        match self {
            AirInstruction::CustomInstruction(i) => i.inputs(),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::inputs(i),
            AirInstruction::BitConstraint(i) => Instruction::<F>::inputs(i),
            AirInstruction::Assign(i) => Instruction::<F>::inputs(i),
            AirInstruction::Cycle(i) => Instruction::<F>::inputs(i),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            AirInstruction::CustomInstruction(i) => i.write(writer, row_index),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::BitConstraint(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Assign(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Cycle(i) => Instruction::<F>::write(i, writer, row_index),
        }
    }

    fn constraint_degree(&self) -> usize {
        match self {
            AirInstruction::CustomInstruction(i) => i.constraint_degree(),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::constraint_degree(i),
            AirInstruction::BitConstraint(i) => Instruction::<F>::constraint_degree(i),
            AirInstruction::Assign(i) => Instruction::<F>::constraint_degree(i),
            AirInstruction::Cycle(i) => Instruction::<F>::constraint_degree(i),
        }
    }

    fn id(&self) -> InstructionId {
        match self {
            AirInstruction::CustomInstruction(i) => i.id(),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::id(i),
            AirInstruction::BitConstraint(i) => Instruction::<F>::id(i),
            AirInstruction::Assign(i) => Instruction::<F>::id(i),
            AirInstruction::Cycle(i) => Instruction::<F>::id(i),
        }
    }
}

impl<F, I> From<I> for AirInstruction<F, I> {
    fn from(instruction: I) -> Self {
        AirInstruction::CustomInstruction(instruction)
    }
}

impl<F, I> AirInstruction<F, I> {
    pub fn write(register: &MemorySlice) -> Self {
        AirInstruction::WriteInstruction(WriteInstruction(*register))
    }

    pub fn bits(register: &MemorySlice) -> Self {
        AirInstruction::BitConstraint(BitConstraint(*register))
    }

    pub fn assign(assignment: AssignInstruction<F>) -> Self {
        AirInstruction::Assign(assignment)
    }

    pub fn cycle(cycle: Cycle<F>) -> Self {
        AirInstruction::Cycle(cycle)
    }
}

impl<F: Field, I: Instruction<F>> PartialEq for AirInstruction<F, I> {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<F: Field, I: Instruction<F>> Eq for AirInstruction<F, I> {}

impl<F: Field, I: Instruction<F>> PartialOrd for AirInstruction<F, I> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id().partial_cmp(&other.id())
    }
}

impl<F: Field, I: Instruction<F>> Ord for AirInstruction<F, I> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id().cmp(&other.id())
    }
}

impl<F: Field, I: Instruction<F>> Hash for AirInstruction<F, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
