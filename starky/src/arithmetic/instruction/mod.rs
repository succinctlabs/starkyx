//! Instruction trait
//!
//! The instruction trait represents the interface of a microcomand in a chip. It is the
//! lowest level of abstraction in the arithmetic module.

pub mod arithmetic_expressions;
pub mod write;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use self::arithmetic_expressions::ArithmeticExpressionSlice;
use super::bool::ConstraintBool;
use super::register::MemorySlice;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

/// Implements methods for writing to the trace of the instruction. This trait was seperated from
/// the `Instruction` trait as we want to use trait objects in the builder.
pub trait InstructionTrace<F: RichField + Extendable<D>, const D: usize> {
    /// Returns a vector of memory slices or contiguous memory regions of the row in the trace that
    /// instruction relies on. These registers must be filled in by the `TraceWriter`.
    fn layout(&self) -> Vec<MemorySlice>;

    /// Assigns the row in the trace according to the `witness_layout`. Usually called by the
    /// `TraceWriter`.
    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.layout()
            .into_iter()
            .fold(0, |local_index, memory_slice| {
                memory_slice.assign(trace_rows, local_index, row, row_index)
            });
    }
}

pub trait Instruction<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Clone + InstructionTrace<F, D>
{
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

/// Constraints don't have any writing data or witness
///
/// These are only used to generate constrains and can therefore
/// be held separate from instructions and all hold the same type.
#[derive(Clone, Debug)]
pub enum EqualityConstraint<F, const D: usize> {
    Bool(ConstraintBool),
    Equal(MemorySlice, MemorySlice),
    ArithmeticConstraint(
        ArithmeticExpressionSlice<F, D>,
        ArithmeticExpressionSlice<F, D>,
    ),
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
                let a_vals = a.packed_generic_vars(&vars);
                let b_vals = b.packed_generic_vars(&vars);
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
                let a_vals = a.ext_circuit_vars(&vars);
                let b_vals = b.ext_circuit_vars(&vars);
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
