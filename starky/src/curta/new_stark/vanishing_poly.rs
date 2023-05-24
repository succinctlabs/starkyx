// use plonky2::field::extension::{Extendable, FieldExtension};
// use plonky2::field::packed::PackedField;
// use plonky2::hash::hash_types::RichField;
// use plonky2::iop::target::Target;
// use plonky2::plonk::circuit_builder::CircuitBuilder;

// use crate::config::StarkConfig;
// use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
// use crate::curta::chip::{ChipStark, StarkParameters};
// use crate::stark::Stark;
// use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

// pub(crate) fn eval_vanishing_poly<F, FE, P, L, const D: usize, const D2: usize>(
//     stark: &ChipStark<L, F, D>,
//     config: &StarkConfig,
//     vars: StarkEvaluationVars<
//         FE,
//         P,
//         { ChipStark::<L, F, D>::COLUMNS },
//         { ChipStark::<L, F, D>::PUBLIC_INPUTS },
//     >,
//     consumer: &mut ConstraintConsumer<P>,
//     betas: &[[F; 3]],
// ) where
//     F: RichField + Extendable<D>,
//     FE: FieldExtension<D2, BaseField = F>,
//     P: PackedField<Scalar = FE>,
//     L: StarkParameters<F, D>,
//     [(); ChipStark::<L, F, D>::COLUMNS]:,
//     [(); ChipStark::<L, F, D>::PUBLIC_INPUTS]:,
// {
//     stark.chip.eval_packed_generic(betas, vars, consumer);
// }

// pub(crate) fn eval_vanishing_poly_circuit<F, L, const D: usize>(
//     builder: &mut CircuitBuilder<F, D>,
//     stark: &ChipStark<L, F, D>,
//     config: &StarkConfig,
//     vars: StarkEvaluationTargets<
//         D,
//         { ChipStark::<L, F, D>::COLUMNS },
//         { ChipStark::<L, F, D>::PUBLIC_INPUTS },
//     >,
//     consumer: &mut RecursiveConstraintConsumer<F, D>,
//     betas: &[[Target; 3]],
// ) where
//     F: RichField + Extendable<D>,
//     L: StarkParameters<F, D>,
//     [(); ChipStark::<L, F, D>::COLUMNS]:,
//     [(); ChipStark::<L, F, D>::PUBLIC_INPUTS]:,
// {
//     stark.chip.eval_ext_circuit(betas, builder, vars, consumer);
// }
