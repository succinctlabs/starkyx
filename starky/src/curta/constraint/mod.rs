//! Equality constraints for the Starky AIR constraint system
//!
//! This module contains the definition of the `Constraint` enum, which is the main
//! abstraction for a constraint for the presentation of AIR used in Starky.
//!

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use self::expression::ConstraintExpression;
use super::air::parser::AirParser;
use super::instruction::Instruction;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub mod arithmetic;
pub mod expression;

/// An abstract representations of a general AIR equality constraint
///
/// An AIR equality constraint is a constraint of the form:
/// P
///
/// Constraints can be applied to first row, last row,
/// transition (which means all but last rows) or all rows.
#[derive(Debug, Clone)]
pub enum Constraint<I, F: RichField + Extendable<D>, const D: usize> {
    /// Enforce the constraint in the first row
    First(ConstraintExpression<I, F, D>),
    /// Enforce the constraint in the last row
    Last(ConstraintExpression<I, F, D>),
    /// Enforce the constraint in all rows but the last
    Transition(ConstraintExpression<I, F, D>),
    /// Enforce the constraint in all rows
    All(ConstraintExpression<I, F, D>),
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize> Constraint<I, F, D> {
    pub fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) {
        match self {
            Constraint::First(constraint) => {
                let vals = constraint.eval(parser);
                for val in vals {
                    parser.constraint_first_row(val);
                }
            }
            Constraint::Last(constraint) => {
                let vals = constraint.eval(parser);
                for val in vals {
                    parser.constraint_last_row(val);
                }
            }
            Constraint::Transition(constraint) => {
                let vals = constraint.eval(parser);
                for val in vals {
                    parser.constraint_transition(val);
                }
            }
            Constraint::All(constraint) => {
                let vals = constraint.eval(parser);
                for val in vals {
                    parser.constraint(val);
                }
            }
        }
    }
}
