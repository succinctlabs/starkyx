#![allow(dead_code)]

use std::sync::mpsc::Sender;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::Register;
use crate::arithmetic::circuit::EmulatedCircuitLayout;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub trait Opcode<F, const D: usize>: 'static + Sized + Send + Sync {
    type Output: Clone + Send + Sync;
    fn generate_trace_row(self) -> (Vec<F>, Self::Output);
}

pub trait OpcodeLayout<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync
{
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

pub struct WriteInputLayout {
    input: Register,
}

impl WriteInputLayout {
    pub const fn new(input: Register) -> Self {
        Self { input }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> OpcodeLayout<F, D> for WriteInputLayout {
    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.input.assign(trace_rows, row, row_index);
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
