use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use self::expression::ArithmeticExpression;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub mod expression;
pub mod instruction;

#[derive(Debug, Clone)]
pub enum ArithmeticConstraint<F: RichField + Extendable<D>, const D: usize> {
    First(ArithmeticExpression<F, D>),
    Last(ArithmeticExpression<F, D>),
    Transition(ArithmeticExpression<F, D>),
    All(ArithmeticExpression<F, D>),
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticConstraint<F, D> {
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
            ArithmeticConstraint::First(constraint) => {
                let vals = constraint.expression.packed_generic(&vars);
                for &val in vals.iter() {
                    yield_constr.constraint_first_row(val);
                }
            }
            ArithmeticConstraint::Last(constraint) => {
                let vals = constraint.expression.packed_generic(&vars);
                for &val in vals.iter() {
                    yield_constr.constraint_last_row(val);
                }
            }
            ArithmeticConstraint::Transition(constraint) => {
                let vals = constraint.expression.packed_generic(&vars);
                for &val in vals.iter() {
                    yield_constr.constraint_transition(val);
                }
            }
            ArithmeticConstraint::All(constraint) => {
                let vals = constraint.expression.packed_generic(&vars);
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
            ArithmeticConstraint::First(constraint) => {
                let vals = constraint.expression.ext_circuit(builder, &vars);
                for &val in vals.iter() {
                    yield_constr.constraint_first_row(builder, val);
                }
            }
            ArithmeticConstraint::Last(constraint) => {
                let vals = constraint.expression.ext_circuit(builder, &vars);
                for &val in vals.iter() {
                    yield_constr.constraint_last_row(builder, val);
                }
            }
            ArithmeticConstraint::Transition(constraint) => {
                let vals = constraint.expression.ext_circuit(builder, &vars);
                for &val in vals.iter() {
                    yield_constr.constraint_transition(builder, val);
                }
            }
            ArithmeticConstraint::All(constraint) => {
                let vals = constraint.expression.ext_circuit(builder, &vars);
                for &val in vals.iter() {
                    yield_constr.constraint(builder, val);
                }
            }
        }
    }
}
