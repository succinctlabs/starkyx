use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::Instruction;
use crate::curta::register::MemorySlice;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

/// A defult instruction set that contains no custom instructions
#[derive(Clone, Debug)]
pub struct EmptyInstructionSet<F, const D: usize> {
    _marker: core::marker::PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for EmptyInstructionSet<F, D> {
    fn assign_row(&self, _trace_rows: &mut [Vec<F>], _row: &mut [F], _row_index: usize) {}

    fn trace_layout(&self) -> Vec<MemorySlice> {
        Vec::new()
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
