//! Instruction trait
//!
//! The instruction trait represents the interface of a microcomand in a chip. It is the
//! lowest level of abstraction in the arithmetic module.

use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::bool::ConstraintBool;
use super::register::{Register, WitnessData};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub trait Instruction<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Clone
{
    //fn generate_trace_row(&self, input: Option<Self::Input>) -> (Vec<F>, Option<Self::Output>);

    fn memory_vec(&self) -> Vec<Register>;

    fn witness_data(&self) -> Option<WitnessData>;

    fn set_witness(&mut self, witness: Register) -> Result<()>;

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
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EqualityConstraint {
    Bool(ConstraintBool),
    Equal(Register, Register),
}

#[derive(Clone, Debug, Copy)]
pub struct WriteInstruction(pub Register);

impl WriteInstruction {
    #[inline]
    pub fn into_register(self) -> Register {
        self.0
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for WriteInstruction {
    fn witness_data(&self) -> Option<WitnessData> {
        None
    }

    fn memory_vec(&self) -> Vec<Register> {
        vec![self.0]
    }

    fn set_witness(&mut self, _witness: Register) -> Result<()> {
        Ok(())
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.0.assign(trace_rows, row, row_index);
    }

    fn packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        _vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        _yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        _yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for EqualityConstraint {
    fn witness_data(&self) -> Option<WitnessData> {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::witness_data(constraint)
            }
            EqualityConstraint::Equal(_a, _b) => None,
        }
    }

    fn memory_vec(&self) -> Vec<Register> {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::memory_vec(constraint)
            }
            EqualityConstraint::Equal(_a, _b) => vec![],
        }
    }

    fn set_witness(&mut self, _witness: Register) -> Result<()> {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::set_witness(constraint, _witness)
            }
            EqualityConstraint::Equal(_a, _b) => Ok(()),
        }
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::assign_row(
                    constraint, trace_rows, row, row_index,
                )
            }
            EqualityConstraint::Equal(_, _) => {}
        }
    }

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
                if let (Register::Local(_, _), Register::Local(_, _)) = (a, b) {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        yield_constr.constraint(a - b);
                    }
                } else {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        yield_constr.constraint_transition(a - b);
                    }
                }
            }
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
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
                if let (Register::Local(_, _), Register::Local(_, _)) = (a, b) {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        let constr = builder.sub_extension(a, b);
                        yield_constr.constraint(builder, constr);
                    }
                } else {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        let constr = builder.sub_extension(a, b);
                        yield_constr.constraint_transition(builder, constr);
                    }
                }
            }
        }
    }
}
