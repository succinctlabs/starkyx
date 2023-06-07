//! Constraints for instructions
//!
//! The instructions constraints can be multiplied by an arithmetic expression. This is usually
//! used for a selector.

use core::ops::{Add, Mul, Sub};
use std::sync::Arc;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::arithmetic::ArithmeticExpression;
use crate::curta::air::parser::AirParser;
use crate::curta::instruction::Instruction;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

/// An abstract representation of a general AIR vanishing polynomial
///
#[derive(Clone, Debug)]
pub enum ConstraintExpression<I, F, const D: usize> {
    /// An empty constraint
    Empty,
    /// The constraints contained in a single instruction
    Instruction(I),
    /// A constraint asserting the arithmetic expression to be zero
    Arithmetic(ArithmeticExpression<F, D>),
    /// A constraint P * Multiplier for a constraint expression P and an arithmetic expression Multiplier
    Mul(
        Arc<ConstraintExpression<I, F, D>>,
        ArithmeticExpression<F, D>,
    ),
    /// Adding the constraints of two expressions Left + Right
    Add(
        Arc<ConstraintExpression<I, F, D>>,
        Arc<ConstraintExpression<I, F, D>>,
    ),
    /// Subtracting the constraints of two expressions Left - Right
    Sub(
        Arc<ConstraintExpression<I, F, D>>,
        Arc<ConstraintExpression<I, F, D>>,
    ),
    /// A constraint for the constraints of left and right
    Union(
        Arc<ConstraintExpression<I, F, D>>,
        Arc<ConstraintExpression<I, F, D>>,
    ),
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize>
    ConstraintExpression<I, F, D>
{
    pub fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) -> Vec<AP::Var> {
        match self {
            ConstraintExpression::Empty => vec![],
            ConstraintExpression::Instruction(instruction) => instruction.eval(parser),
            ConstraintExpression::Arithmetic(expr) => expr.expression.eval(parser),
            ConstraintExpression::Mul(instruction, multiplier) => {
                let vals = instruction.eval(parser);
                assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                let mult = multiplier.expression.eval(parser)[0];
                vals.iter().map(|val| parser.mul(*val, mult)).collect()
            }
            ConstraintExpression::Add(left, right) => {
                let left = left.eval(parser);
                let right = right.eval(parser);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| parser.add(*l, *r))
                    .collect()
            }
            ConstraintExpression::Sub(left, right) => {
                let left = left.eval(parser);
                let right = right.eval(parser);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| parser.sub(*l, *r))
                    .collect()
            }
            ConstraintExpression::Union(left, right) => {
                let mut constraints = left.eval(parser);
                constraints.extend(right.eval(parser));
                constraints
            }
        }
    }

    pub fn instructions(&self) -> Vec<I> {
        match self {
            ConstraintExpression::Empty => vec![],
            ConstraintExpression::Instruction(instruction) => vec![instruction.clone()],
            ConstraintExpression::Arithmetic(_) => vec![],
            ConstraintExpression::Mul(instruction, _) => instruction.instructions(),
            ConstraintExpression::Add(left, right) => {
                let mut instructions = left.instructions();
                instructions.extend(right.instructions());
                instructions
            }
            ConstraintExpression::Sub(left, right) => {
                let mut instructions = left.instructions();
                instructions.extend(right.instructions());
                instructions
            }
            ConstraintExpression::Union(left, right) => {
                let mut instructions = left.instructions();
                instructions.extend(right.instructions());
                instructions
            }
        }
    }

    pub fn union(self, other: Self) -> Self {
        ConstraintExpression::Union(Arc::new(self), Arc::new(other))
    }

    pub fn union_from_iter(expressions: impl Iterator<Item = Self>) -> Self {
        expressions.fold(ConstraintExpression::Empty, |acc, expr| acc.union(expr))
    }

    pub fn union_all(expressions: impl IntoIterator<Item = Self>) -> Self {
        Self::union_from_iter(expressions.into_iter())
    }
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize>
    From<ArithmeticExpression<F, D>> for ConstraintExpression<I, F, D>
{
    fn from(expr: ArithmeticExpression<F, D>) -> Self {
        ConstraintExpression::Arithmetic(expr)
    }
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize> Add
    for ConstraintExpression<I, F, D>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ConstraintExpression::Add(Arc::new(self), Arc::new(rhs))
    }
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize> Sub
    for ConstraintExpression<I, F, D>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        ConstraintExpression::Sub(Arc::new(self), Arc::new(rhs))
    }
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize>
    Mul<ArithmeticExpression<F, D>> for ConstraintExpression<I, F, D>
{
    type Output = Self;

    fn mul(self, rhs: ArithmeticExpression<F, D>) -> Self::Output {
        ConstraintExpression::Mul(Arc::new(self), rhs)
    }
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize> Mul<F>
    for ConstraintExpression<I, F, D>
{
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        ConstraintExpression::Mul(Arc::new(self), ArithmeticExpression::from_constant(rhs))
    }
}
