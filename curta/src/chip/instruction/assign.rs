use std::collections::HashSet;

use itertools::Itertools;

use super::Instruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;
use crate::trace::writer::TraceWriter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssignType {
    First,
    Last,
    Transition,
    All,
}

#[derive(Clone, Debug)]
pub struct AssignInstruction<F> {
    pub source: ArithmeticExpression<F>,
    pub target: MemorySlice,
    pub kind: AssignType,
}

impl<F> AssignInstruction<F> {
    pub fn new(source: ArithmeticExpression<F>, target: MemorySlice, kind: AssignType) -> Self {
        Self {
            source,
            target,
            kind,
        }
    }
}

impl<F: Field, AP: AirParser<Field = F>> AirConstraint<AP> for AssignInstruction<F> {
    fn eval(&self, parser: &mut AP) {
        let expression = self.source.eval(parser);
        let targets = self.target.eval_slice(parser).to_vec();

        match self.kind {
            AssignType::First => {
                for (target, value) in targets.into_iter().zip_eq(expression) {
                    let difference = parser.sub(target, value);
                    parser.constraint_first_row(difference);
                }
            }
            AssignType::Last => {
                for (target, value) in targets.into_iter().zip_eq(expression) {
                    let difference = parser.sub(target, value);
                    parser.constraint_last_row(difference);
                }
            }
            AssignType::Transition => {
                for (target, value) in targets.into_iter().zip_eq(expression) {
                    let difference = parser.sub(target, value);
                    parser.constraint_transition(difference);
                }
            }
            AssignType::All => {
                for (target, value) in targets.into_iter().zip_eq(expression) {
                    let difference = parser.sub(target, value);
                    parser.constraint(difference);
                }
            }
        }
    }
}

impl<F: Field> Instruction<F> for AssignInstruction<F> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![self.target]
    }

    fn inputs(&self) -> HashSet<MemorySlice> {
        self.source.registers().into_iter().collect()
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self.kind {
            AssignType::First => {
                if row_index == 0 {
                    let value = writer.read_expression(&self.source, row_index);
                    writer.write(&self.target, &value, 0);
                }
            }
            AssignType::Last => {
                if row_index == writer.height() - 1 {
                    let value = writer.read_expression(&self.source, row_index);
                    writer.write(&self.target, &value, row_index);
                }
            }
            AssignType::Transition => {
                if row_index < writer.height() - 1 {
                    let value = writer.read_expression(&self.source, row_index);
                    writer.write(&self.target, &value, row_index);
                }
            }
            AssignType::All => {
                let value = writer.read_expression(&self.source, row_index);
                writer.write(&self.target, &value, row_index);
            }
        }
    }
}
