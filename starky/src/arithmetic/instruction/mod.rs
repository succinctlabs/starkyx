//! Instruction trait
//!
//! The instruction trait represents the interface of a microcomand in a chip. It is the
//! lowest level of abstraction in the arithmetic module.

pub mod arithmetic_expressions;
pub mod write;

use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use self::arithmetic_expressions::ArithmeticExpression;
use super::bool::ConstraintBool;
use super::register::MemorySlice;
use crate::arithmetic::register::WitnessData;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub trait Instruction<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Clone
{
    //fn generate_trace_row(&self, input: Option<Self::Input>) -> (Vec<F>, Option<Self::Output>);

    fn memory_vec(&self) -> Vec<MemorySlice>;

    fn witness_data(&self) -> Option<WitnessData>;

    fn set_witness(&mut self, witness: MemorySlice) -> Result<()>;

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize);

    fn packed_generic_constraints<
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
        P: PackedField<Scalar = FE>;

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    );
}

/// Constraints don't have any writing data or witness
///
/// These are only used to generate constrains and can therefore
/// be held separate from instructions and all hold the same type.
#[derive(Clone, Debug)]
pub enum EqualityConstraint<F, const D: usize> {
    Bool(ConstraintBool),
    Equal(MemorySlice, MemorySlice),
    ArithmeticConstraint(ArithmeticExpression<F, D>, ArithmeticExpression<F, D>),
}

impl<F: RichField + Extendable<D>, const D: usize> EqualityConstraint<F, D> {
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
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::packed_generic_constraints(
                    constraint,
                    vars,
                    yield_constr,
                )
            }
            EqualityConstraint::Equal(a, b) => {
                let a_vals = a.packed_entries_slice(&vars);
                let b_vals = b.packed_entries_slice(&vars);
                if let (MemorySlice::Local(_, _), MemorySlice::Local(_, _)) = (a, b) {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        yield_constr.constraint(a - b);
                    }
                } else {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        yield_constr.constraint_transition(a - b);
                    }
                }
            }
            EqualityConstraint::ArithmeticConstraint(left, right) => {
                let left_vals = left.packed_generic(&vars);
                let right_vals = right.packed_generic(&vars);
                for (a, b) in left_vals.iter().zip(right_vals.iter()) {
                    yield_constr.constraint_transition(*a - *b);
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
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::ext_circuit_constraints(
                    constraint,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            EqualityConstraint::Equal(a, b) => {
                let a_vals = a.evaluation_targets(&vars);
                let b_vals = b.evaluation_targets(&vars);
                if let (MemorySlice::Local(_, _), MemorySlice::Local(_, _)) = (a, b) {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        let constr = builder.sub_extension(a, b);
                        yield_constr.constraint_transition(builder, constr);
                    }
                } else {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        let constr = builder.sub_extension(a, b);
                        yield_constr.constraint_transition(builder, constr);
                    }
                }
            }
            EqualityConstraint::ArithmeticConstraint(left, right) => {
                let left_vals = left.ext_circuit(builder, &vars);
                let right_vals = right.ext_circuit(builder, &vars);
                for (a, b) in left_vals.iter().zip(right_vals.iter()) {
                    let constr = builder.sub_extension(*a, *b);
                    yield_constr.constraint_transition(builder, constr);
                }
            }
        }
    }
}
