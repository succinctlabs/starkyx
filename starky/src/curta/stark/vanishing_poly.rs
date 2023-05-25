use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::config::StarkConfig;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::curta::chip::{ChipStark, StarkParameters};
use crate::permutation::{
    eval_permutation_checks, eval_permutation_checks_circuit, PermutationCheckDataTarget,
    PermutationCheckVars,
};
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub(crate) fn eval_vanishing_poly<F, FE, P, L, const D: usize, const D2: usize>(
    stark: &ChipStark<L, F, D>,
    config: &StarkConfig,
    vars: StarkEvaluationVars<
        FE,
        P,
        { ChipStark::<L, F, D>::COLUMNS },
        { ChipStark::<L, F, D>::PUBLIC_INPUTS },
    >,
    permutation_data: Option<PermutationCheckVars<F, FE, P, D2>>,
    consumer: &mut ConstraintConsumer<P>,
    betas: &[F],
) where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
    L: StarkParameters<F, D>,
    [(); ChipStark::<L, F, D>::COLUMNS]:,
    [(); ChipStark::<L, F, D>::PUBLIC_INPUTS]:,
{
    stark.chip.eval_packed_generic(betas, vars, consumer);
    if let Some(permutation_data) = permutation_data {
        eval_permutation_checks::<F, FE, P, ChipStark<L, F, D>, D, D2>(
            stark,
            config,
            vars,
            permutation_data,
            consumer,
        );
    }
}

pub(crate) fn eval_vanishing_poly_circuit<F, L, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &ChipStark<L, F, D>,
    config: &StarkConfig,
    vars: StarkEvaluationTargets<
        D,
        { ChipStark::<L, F, D>::COLUMNS },
        { ChipStark::<L, F, D>::PUBLIC_INPUTS },
    >,
    permutation_data: Option<PermutationCheckDataTarget<D>>,
    consumer: &mut RecursiveConstraintConsumer<F, D>,
    betas: &[Target],
) where
    F: RichField + Extendable<D>,
    L: StarkParameters<F, D>,
    [(); ChipStark::<L, F, D>::COLUMNS]:,
    [(); ChipStark::<L, F, D>::PUBLIC_INPUTS]:,
{
    stark.chip.eval_ext_circuit(betas, builder, vars, consumer);
    if let Some(permutation_data) = permutation_data {
        eval_permutation_checks_circuit::<F, ChipStark<L, F, D>, D>(
            builder,
            stark,
            config,
            vars,
            permutation_data,
            consumer,
        );
    }
}
