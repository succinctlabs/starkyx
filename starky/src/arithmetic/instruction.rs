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

#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypeConstraint {
    Bool(ConstraintBool),
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

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for TypeConstraint {
    fn witness_data(&self) -> Option<WitnessData> {
        match self {
            TypeConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::witness_data(constraint)
            }
        }
    }

    fn memory_vec(&self) -> Vec<Register> {
        match self {
            TypeConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::memory_vec(constraint)
            }
        }
    }

    fn set_witness(&mut self, _witness: Register) -> Result<()> {
        match self {
            TypeConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::set_witness(constraint, _witness)
            }
        }
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            TypeConstraint::Bool(constraint) => <ConstraintBool as Instruction<F, D>>::assign_row(
                constraint, trace_rows, row, row_index,
            ),
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
            TypeConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::packed_generic_constraints(
                    constraint,
                    vars,
                    yield_constr,
                )
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
            TypeConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::ext_circuit_constraints(
                    constraint,
                    builder,
                    vars,
                    yield_constr,
                )
            }
        }
    }
}