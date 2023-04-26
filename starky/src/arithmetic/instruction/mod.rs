//! Instruction trait
//!
//! The instruction trait represents the interface of a microcomand in a chip. It is the
//! lowest level of abstraction in the arithmetic module.

pub mod arithmetic_expressions;
pub mod empty;
mod set;
pub mod write;

pub use empty::EmptyInstructionSet;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
pub use set::{FromInstructionSet, InstructionSet};

use self::arithmetic_expressions::ArithmeticExpression;
use super::field::{
    FpAddInstruction, FpInnerProductInstruction, FpMulConstInstruction, FpMulInstruction,
};
use super::register::MemorySlice;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub trait Instruction<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Clone
{
    /// Returns a vector of memory slices or contiguous memory regions of the row in the trace that
    /// instruction relies on. These registers must be filled in by the `TraceWriter`.
    fn trace_layout(&self) -> Vec<MemorySlice>;

    /// Assigns the row in the trace according to the `witness_layout`. Usually called by the
    /// `TraceWriter`.
    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.trace_layout()
            .into_iter()
            .fold(0, |local_index, memory_slice| {
                memory_slice.assign(trace_rows, local_index, row, row_index)
            });
    }

    /// Constrains the instruction properly within the STARK by using the `ConstraintConsumer`.
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

    /// Constrains the instruction properly within Plonky2 by using the `RecursiveConstraintConsumer`.
    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    );
}

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
