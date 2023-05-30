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
use crate::curta::instruction::Instruction;
use crate::curta::new_stark::vars as new_vars;
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
    pub fn packed_generic<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<P>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            ConstraintExpression::Empty => vec![],
            ConstraintExpression::Instruction(instruction) => instruction.packed_generic(vars),
            ConstraintExpression::Arithmetic(expr) => expr.expression.packed_generic(vars),
            ConstraintExpression::Mul(instruction, multiplier) => {
                let vals = instruction.packed_generic(vars);
                assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                let mult = multiplier.expression.packed_generic(vars)[0];
                vals.iter().map(|&val| val * mult).collect()
            }
            ConstraintExpression::Add(left, right) => {
                let left = left.packed_generic(vars);
                let right = right.packed_generic(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r + *l)
                    .collect()
            }
            ConstraintExpression::Sub(left, right) => {
                let left = left.packed_generic(vars);
                let right = right.packed_generic(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r - *l)
                    .collect()
            }
            ConstraintExpression::Union(left, right) => {
                let mut constraints = left.packed_generic(vars);
                constraints.extend(right.packed_generic(vars));
                constraints
            }
        }
    }

    pub fn ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<ExtensionTarget<D>> {
        match self {
            ConstraintExpression::Empty => vec![],
            ConstraintExpression::Instruction(instruction) => {
                instruction.ext_circuit(builder, vars)
            }
            ConstraintExpression::Arithmetic(expr) => expr.expression.ext_circuit(builder, vars),
            ConstraintExpression::Mul(instruction, multiplier) => {
                let vals = instruction.ext_circuit(builder, vars);
                assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                let mult = multiplier.expression.ext_circuit(builder, vars)[0];
                vals.iter()
                    .map(|val| builder.mul_extension(*val, mult))
                    .collect()
            }
            ConstraintExpression::Add(left, right) => {
                let left = left.ext_circuit(builder, vars);
                let right = right.ext_circuit(builder, vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| builder.add_extension(*l, *r))
                    .collect()
            }
            ConstraintExpression::Sub(left, right) => {
                let left = left.ext_circuit(builder, vars);
                let right = right.ext_circuit(builder, vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| builder.sub_extension(*l, *r))
                    .collect()
            }
            ConstraintExpression::Union(left, right) => {
                let mut constraints = left.ext_circuit(builder, vars);
                constraints.extend(right.ext_circuit(builder, vars));
                constraints
            }
        }
    }

    pub fn packed_generic_new<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
    >(
        &self,
        vars: new_vars::StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }, { CHALLENGES }>,
    ) -> Vec<P>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            ConstraintExpression::Empty => vec![],
            ConstraintExpression::Instruction(instruction) => instruction.packed_generic_new(vars),
            ConstraintExpression::Arithmetic(expr) => expr.expression.packed_generic_new(vars),
            ConstraintExpression::Mul(instruction, multiplier) => {
                let vals = instruction.packed_generic_new(vars);
                assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                let mult = multiplier.expression.packed_generic_new(vars)[0];
                vals.iter().map(|&val| val * mult).collect()
            }
            ConstraintExpression::Add(left, right) => {
                let left = left.packed_generic_new(vars);
                let right = right.packed_generic_new(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r + *l)
                    .collect()
            }
            ConstraintExpression::Sub(left, right) => {
                let left = left.packed_generic_new(vars);
                let right = right.packed_generic_new(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r - *l)
                    .collect()
            }
            ConstraintExpression::Union(left, right) => {
                let mut constraints = left.packed_generic_new(vars);
                constraints.extend(right.packed_generic_new(vars));
                constraints
            }
        }
    }

    pub fn ext_circuit_new<
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: new_vars::StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }, { CHALLENGES }>,
    ) -> Vec<ExtensionTarget<D>> {
        match self {
            ConstraintExpression::Empty => vec![],
            ConstraintExpression::Instruction(instruction) => {
                instruction.ext_circuit_new(builder, vars)
            }
            ConstraintExpression::Arithmetic(expr) => {
                expr.expression.ext_circuit_new(builder, vars)
            }
            ConstraintExpression::Mul(instruction, multiplier) => {
                let vals = instruction.ext_circuit_new(builder, vars);
                assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                let mult = multiplier.expression.ext_circuit_new(builder, vars)[0];
                vals.iter()
                    .map(|val| builder.mul_extension(*val, mult))
                    .collect()
            }
            ConstraintExpression::Add(left, right) => {
                let left = left.ext_circuit_new(builder, vars);
                let right = right.ext_circuit_new(builder, vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| builder.add_extension(*l, *r))
                    .collect()
            }
            ConstraintExpression::Sub(left, right) => {
                let left = left.ext_circuit_new(builder, vars);
                let right = right.ext_circuit_new(builder, vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| builder.sub_extension(*l, *r))
                    .collect()
            }
            ConstraintExpression::Union(left, right) => {
                let mut constraints = left.ext_circuit_new(builder, vars);
                constraints.extend(right.ext_circuit_new(builder, vars));
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
