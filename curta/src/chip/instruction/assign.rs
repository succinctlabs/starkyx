use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::Instruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AssignType {
    First,
    Last,
    Transition,
    All,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        match self.kind {
            AssignType::First => {
                if row_index == 0 {
                    let value = writer.read_expression(&self.source, row_index);
                    writer.write_slice(&self.target, &value, 0);
                }
            }
            AssignType::Last => {
                if row_index == writer.height() - 1 {
                    let value = writer.read_expression(&self.source, row_index);
                    writer.write_slice(&self.target, &value, row_index);
                }
            }
            AssignType::Transition => {
                if row_index < writer.height() - 1 {
                    let value = writer.read_expression(&self.source, row_index);
                    writer.write_slice(&self.target, &value, row_index);
                }
            }
            AssignType::All => {
                let value = writer.read_expression(&self.source, row_index);
                writer.write_slice(&self.target, &value, row_index);
            }
        }
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let row_index = writer.row_index();
        let height = writer.height();
        match (self.kind, row_index) {
            (AssignType::First, Some(0)) => {
                let value = writer.read_expression(&self.source);
                writer.write_slice(&self.target, &value);
            }
            (AssignType::Last, Some(r)) if r == height - 1 => {
                let value = writer.read_expression(&self.source);
                writer.write_slice(&self.target, &value);
            }
            (AssignType::Transition, Some(r)) if r < height - 1 => {
                let value = writer.read_expression(&self.source);
                writer.write_slice(&self.target, &value);
            }
            (AssignType::All, _) => {
                let value = writer.read_expression(&self.source);
                writer.write_slice(&self.target, &value);
            }
            _ => {}
        }
    }
}
