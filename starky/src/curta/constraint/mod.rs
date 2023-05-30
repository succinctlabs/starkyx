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
use super::instruction::Instruction;
use crate::curta::new_stark::vars as new_vars;
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
    pub fn packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            Constraint::First(constraint) => {
                let vals = constraint.packed_generic(vars);
                for &val in vals.iter() {
                    yield_constr.constraint_first_row(val);
                }
            }
            Constraint::Last(constraint) => {
                let vals = constraint.packed_generic(vars);
                for &val in vals.iter() {
                    yield_constr.constraint_last_row(val);
                }
            }
            Constraint::Transition(constraint) => {
                let vals = constraint.packed_generic(vars);
                for &val in vals.iter() {
                    yield_constr.constraint_transition(val);
                }
            }
            Constraint::All(constraint) => {
                let vals = constraint.packed_generic(vars);
                for &val in vals.iter() {
                    yield_constr.constraint(val);
                }
            }
        }
    }

    pub fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            Constraint::First(constraint) => {
                let vals = constraint.ext_circuit(builder, vars);
                for &val in vals.iter() {
                    yield_constr.constraint_first_row(builder, val);
                }
            }
            Constraint::Last(constraint) => {
                let vals = constraint.ext_circuit(builder, vars);
                for &val in vals.iter() {
                    yield_constr.constraint_last_row(builder, val);
                }
            }
            Constraint::Transition(constraint) => {
                let vals = constraint.ext_circuit(builder, vars);
                for &val in vals.iter() {
                    yield_constr.constraint_transition(builder, val);
                }
            }
            Constraint::All(constraint) => {
                let vals = constraint.ext_circuit(builder, vars);
                for &val in vals.iter() {
                    yield_constr.constraint(builder, val);
                }
            }
        }
    }

    pub fn packed_generic_constraints_new<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
    >(
        &self,
        vars: new_vars::StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }, { CHALLENGES }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            Constraint::First(constraint) => {
                let vals = constraint.packed_generic_new(vars);
                for &val in vals.iter() {
                    yield_constr.constraint_first_row(val);
                }
            }
            Constraint::Last(constraint) => {
                let vals = constraint.packed_generic_new(vars);
                for &val in vals.iter() {
                    yield_constr.constraint_last_row(val);
                }
            }
            Constraint::Transition(constraint) => {
                let vals = constraint.packed_generic_new(vars);
                for &val in vals.iter() {
                    yield_constr.constraint_transition(val);
                }
            }
            Constraint::All(constraint) => {
                let vals = constraint.packed_generic_new(vars);
                for &val in vals.iter() {
                    yield_constr.constraint(val);
                }
            }
        }
    }

    pub fn ext_circuit_constraints_new<
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: new_vars::StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }, { CHALLENGES }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            Constraint::First(constraint) => {
                let vals = constraint.ext_circuit_new(builder, vars);
                for &val in vals.iter() {
                    yield_constr.constraint_first_row(builder, val);
                }
            }
            Constraint::Last(constraint) => {
                let vals = constraint.ext_circuit_new(builder, vars);
                for &val in vals.iter() {
                    yield_constr.constraint_last_row(builder, val);
                }
            }
            Constraint::Transition(constraint) => {
                let vals = constraint.ext_circuit_new(builder, vars);
                for &val in vals.iter() {
                    yield_constr.constraint_transition(builder, val);
                }
            }
            Constraint::All(constraint) => {
                let vals = constraint.ext_circuit_new(builder, vars);
                for &val in vals.iter() {
                    yield_constr.constraint(builder, val);
                }
            }
        }
    }
}
