use alloc::sync::Arc;

use log::debug;
use serde::{Deserialize, Serialize};

use super::assign::AssignInstruction;
use super::bit::BitConstraint;
use super::clock::ClockInstruction;
use super::cycle::{Cycle, ProcessIdInstruction};
use super::Instruction;
use crate::air::parser::{AirParser, MulParser};
use crate::air::AirConstraint;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::bool::SelectInstruction;
use crate::chip::memory::instruction::MemoryInstruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AirInstruction<F, I> {
    CustomInstruction(I),
    BitConstraint(BitConstraint),
    Assign(AssignInstruction<F>),
    Select(SelectInstruction),
    Cycle(Cycle<F>),
    Clock(ClockInstruction),
    ProcessId(ProcessIdInstruction),
    Filtered(ArithmeticExpression<F>, Arc<Self>),
    Mem(MemoryInstruction<F>),
    Watch(String, ArrayRegister<ElementRegister>),
}

impl<F: Field, AP: AirParser<Field = F>, I> AirConstraint<AP> for AirInstruction<F, I>
where
    I: AirConstraint<AP> + for<'a> AirConstraint<MulParser<'a, AP>>,
{
    fn eval(&self, parser: &mut AP) {
        match self {
            AirInstruction::CustomInstruction(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::BitConstraint(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Assign(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Select(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Cycle(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Clock(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::ProcessId(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Filtered(expression, instr) => {
                assert_eq!(
                    expression.size, 1,
                    "Expression multiplying instruction must be of size 1"
                );
                let filter = expression.eval(parser)[0];
                let mut mul_parser = MulParser::new(parser, filter);
                match instr.as_ref() {
                    AirInstruction::CustomInstruction(i) => i.eval(&mut mul_parser),
                    AirInstruction::BitConstraint(i) => i.eval(&mut mul_parser),
                    AirInstruction::Assign(i) => i.eval(&mut mul_parser),
                    AirInstruction::Cycle(i) => i.eval(&mut mul_parser),
                    _ => unreachable!("Instructions cannot be filtered twice"),
                }
            }
            AirInstruction::Mem(i) => AirConstraint::<AP>::eval(i, parser),
            AirInstruction::Watch(_, _) => {}
        }
    }
}

impl<F: Field, I: Instruction<F>> Instruction<F> for AirInstruction<F, I> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self {
            AirInstruction::CustomInstruction(i) => i.write(writer, row_index),
            AirInstruction::BitConstraint(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Select(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Assign(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Cycle(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Clock(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::ProcessId(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Filtered(expression, i) => {
                let filter = writer.read_expression(expression, row_index)[0];
                if filter == F::ONE {
                    i.write(writer, row_index)
                }
            }
            AirInstruction::Mem(i) => Instruction::<F>::write(i, writer, row_index),
            AirInstruction::Watch(name, register) => {
                let value = writer.read_vec(register, row_index);
                debug!("row {}: , {}: {:?}", row_index, name, value);
            }
        }
    }

    fn write_to_air(&self, writer: &mut impl crate::chip::trace::writer::AirWriter<Field = F>) {
        match self {
            AirInstruction::CustomInstruction(i) => i.write_to_air(writer),
            AirInstruction::BitConstraint(i) => i.write_to_air(writer),
            AirInstruction::Select(i) => i.write_to_air(writer),
            AirInstruction::Assign(i) => i.write_to_air(writer),
            AirInstruction::Cycle(i) => i.write_to_air(writer),
            AirInstruction::Clock(i) => i.write_to_air(writer),
            AirInstruction::ProcessId(i) => i.write_to_air(writer),
            AirInstruction::Filtered(expression, i) => {
                let filter = writer.read_expression(expression)[0];
                if filter == F::ONE {
                    i.write_to_air(writer)
                }
            }
            AirInstruction::Mem(i) => i.write_to_air(writer),
            AirInstruction::Watch(name, register) => {
                let value = writer.read_vec(register);
                let row_index = writer.row_index();
                if let Some(index) = row_index {
                    debug!("row {}: , {}: {:?}", index, name, value);
                } else {
                    debug!("{}: {:?}", name, value);
                }
            }
        }
    }
}

impl<F, I> From<I> for AirInstruction<F, I> {
    fn from(instruction: I) -> Self {
        AirInstruction::CustomInstruction(instruction)
    }
}

impl<F, I> AirInstruction<F, I> {
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

    pub fn mem(instruction: MemoryInstruction<F>) -> Self {
        AirInstruction::Mem(instruction)
    }

    pub fn clock(instruction: ClockInstruction) -> Self {
        AirInstruction::Clock(instruction)
    }
}
