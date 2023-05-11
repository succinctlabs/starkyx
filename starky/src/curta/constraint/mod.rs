use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use self::expression::ConstraintExpression;
use super::instruction::Instruction;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub mod arithmetic;
pub mod expression;

// #[derive(Debug, Clone)]
// pub enum ArithmeticConstraint<F: RichField + Extendable<D>, const D: usize> {
//     First(ArithmeticExpression<F, D>),
//     Last(ArithmeticExpression<F, D>),
//     Transition(ArithmeticExpression<F, D>),
//     All(ArithmeticExpression<F, D>),
// }

// impl<F: RichField + Extendable<D>, const D: usize> ArithmeticConstraint<F, D> {
//     pub fn packed_generic_constraints<
//         FE,
//         P,
//         const D2: usize,
//         const COLUMNS: usize,
//         const PUBLIC_INPUTS: usize,
//     >(
//         &self,
//         vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
//         yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
//     ) where
//         FE: FieldExtension<D2, BaseField = F>,
//         P: PackedField<Scalar = FE>,
//     {
//         match self {
//             ArithmeticConstraint::First(constraint) => {
//                 let vals = constraint.expression.packed_generic(vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_first_row(val);
//                 }
//             }
//             ArithmeticConstraint::Last(constraint) => {
//                 let vals = constraint.expression.packed_generic(vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_last_row(val);
//                 }
//             }
//             ArithmeticConstraint::Transition(constraint) => {
//                 let vals = constraint.expression.packed_generic(vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_transition(val);
//                 }
//             }
//             ArithmeticConstraint::All(constraint) => {
//                 let vals = constraint.expression.packed_generic(vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint(val);
//                 }
//             }
//         }
//     }

//     pub fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
//         &self,
//         builder: &mut CircuitBuilder<F, D>,
//         vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
//         yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
//     ) {
//         match self {
//             ArithmeticConstraint::First(constraint) => {
//                 let vals = constraint.expression.ext_circuit(builder, vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_first_row(builder, val);
//                 }
//             }
//             ArithmeticConstraint::Last(constraint) => {
//                 let vals = constraint.expression.ext_circuit(builder, vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_last_row(builder, val);
//                 }
//             }
//             ArithmeticConstraint::Transition(constraint) => {
//                 let vals = constraint.expression.ext_circuit(builder, vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_transition(builder, val);
//                 }
//             }
//             ArithmeticConstraint::All(constraint) => {
//                 let vals = constraint.expression.ext_circuit(builder, vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint(builder, val);
//                 }
//             }
//         }
//     }
// }

#[derive(Debug, Clone)]
pub enum Constraint<I, F: RichField + Extendable<D>, const D: usize> {
    First(ConstraintExpression<I, F, D>),
    Last(ConstraintExpression<I, F, D>),
    Transition(ConstraintExpression<I, F, D>),
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
}
