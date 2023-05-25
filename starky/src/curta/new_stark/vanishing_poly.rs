use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use super::Stark;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn eval_vanishing_poly<F, FE, P, S, const D: usize, const D2: usize, const R: usize>(
    stark: &S,
    vars: StarkEvaluationVars<FE, P, { S::COLUMNS }, { S::PUBLIC_INPUTS }, { S::CHALLENGES }>,
    consumer: &mut ConstraintConsumer<P>,
) where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
    S: Stark<F, D, R>,
    [(); S::COLUMNS]:,
    [(); S::PUBLIC_INPUTS]:,
    [(); S::CHALLENGES]:,
{
    stark.eval_packed_generic(vars, consumer);
}

pub(crate) fn eval_vanishing_poly_circuit<F, S, const D: usize, const R: usize>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &S,
    vars: StarkEvaluationTargets<D, { S::COLUMNS }, { S::PUBLIC_INPUTS }, { S::CHALLENGES }>,
    consumer: &mut RecursiveConstraintConsumer<F, D>,
) where
    F: RichField + Extendable<D>,
    S: Stark<F, D, R>,
    [(); S::COLUMNS]:,
    [(); S::PUBLIC_INPUTS]:,
    [(); S::CHALLENGES]:,
{
    stark.eval_ext_circuit(builder, vars, consumer);
}
