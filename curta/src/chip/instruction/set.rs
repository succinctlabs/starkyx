use alloc::sync::Arc;
use core::hash::{Hash, Hasher};

use super::assign::AssignInstruction;
use super::bit::BitConstraint;
use super::cycle::Cycle;
use super::write::WriteInstruction;
use super::{Instruction, InstructionId};
use crate::air::parser::{AirParser, MulParser};
use crate::air::AirConstraint;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum AirInstruction<F, I> {
    CustomInstruction(I),
    WriteInstruction(WriteInstruction),
    BitConstraint(BitConstraint),
    Assign(AssignInstruction<F>),
    Cycle(Cycle<F>),
    Filtered(ArithmeticExpression<F>, Arc<Self>),
}

impl<F: Field, AP: AirParser<Field = F>, I> AirConstraint<AP> for AirInstruction<F, I>
where
    I: AirConstraint<AP> + for<'a> AirConstraint<MulParser<'a, AP>>,
{
    fn eval(&self, parser: &mut AP) {
        match self {
            AirInstruction::CustomInstruction(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::WriteInstruction(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::BitConstraint(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Assign(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Cycle(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Filtered(expression, instr) => {
                assert_eq!(
                    expression.size, 1,
                    "Expression multiplying instruction must be of size 1"
                );
                let filter = expression.eval(parser)[0];
                let mut mul_parser = MulParser::new(parser, filter);
                match instr.as_ref() {
                    AirInstruction::CustomInstruction(i) => i.eval(&mut mul_parser),
                    AirInstruction::WriteInstruction(i) => i.eval(&mut mul_parser),
                    AirInstruction::BitConstraint(i) => i.eval(&mut mul_parser),
                    AirInstruction::Assign(i) => i.eval(&mut mul_parser),
                    AirInstruction::Cycle(i) => i.eval(&mut mul_parser),
                    _ => unreachable!("Instructions cannot be filtered twice"),
                }
            }
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
            AirInstruction::Filtered(_, i) => i.trace_layout(),
        }
    }

    fn inputs(&self) -> Vec<MemorySlice> {
        match self {
            AirInstruction::CustomInstruction(i) => i.inputs(),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::inputs(i),
            AirInstruction::BitConstraint(i) => Instruction::<F>::inputs(i),
            AirInstruction::Assign(i) => Instruction::<F>::inputs(i),
            AirInstruction::Cycle(i) => Instruction::<F>::inputs(i),
            AirInstruction::Filtered(_, i) => i.inputs(),
        }
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            AirInstruction::CustomInstruction(i) => i.write(writer, row_index),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::BitConstraint(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Assign(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Cycle(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Filtered(expression, i) => {
                let filter = writer.read_expression(expression, row_index)[0];
                if filter == F::ONE {
                    i.write(writer, row_index)
                }
            }
        }
    }

    fn constraint_degree(&self) -> usize {
        match self {
            AirInstruction::CustomInstruction(i) => i.constraint_degree(),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::constraint_degree(i),
            AirInstruction::BitConstraint(i) => Instruction::<F>::constraint_degree(i),
            AirInstruction::Assign(i) => Instruction::<F>::constraint_degree(i),
            AirInstruction::Cycle(i) => Instruction::<F>::constraint_degree(i),
            AirInstruction::Filtered(_, i) => i.constraint_degree() + 1,
        }
    }

    fn id(&self) -> InstructionId {
        match self {
            AirInstruction::CustomInstruction(i) => i.id(),
            AirInstruction::WriteInstruction(i) => Instruction::<F>::id(i),
            AirInstruction::BitConstraint(i) => Instruction::<F>::id(i),
            AirInstruction::Assign(i) => Instruction::<F>::id(i),
            AirInstruction::Cycle(i) => Instruction::<F>::id(i),
            AirInstruction::Filtered(_, i) => i.id(),
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

    pub fn as_filtered(self, filter: ArithmeticExpression<F>) -> Self {
        AirInstruction::Filtered(filter, Arc::new(self))
    }
}

impl<F: Field, I: Instruction<F>> Hash for AirInstruction<F, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
