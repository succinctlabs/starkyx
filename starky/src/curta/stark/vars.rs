use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::iop::ext_target::ExtensionTarget;

#[derive(Debug, Copy, Clone)]
pub struct StarkEvaluationVars<
    'a,
    F,
    P,
    const COLUMNS: usize,
    const PUBLIC_INPUTS: usize,
    const CHALLENGES: usize,
> where
    F: Field,
    P: PackedField<Scalar = F>,
{
    pub local_values: &'a [P; COLUMNS],
    pub next_values: &'a [P; COLUMNS],
    pub public_inputs: &'a [P::Scalar; PUBLIC_INPUTS],
    pub challenges: &'a [P::Scalar; CHALLENGES],
}

#[derive(Debug, Copy, Clone)]
pub struct StarkEvaluationTargets<
    'a,
    const D: usize,
    const COLUMNS: usize,
    const PUBLIC_INPUTS: usize,
    const CHALLENGES: usize,
> {
    pub local_values: &'a [ExtensionTarget<D>; COLUMNS],
    pub next_values: &'a [ExtensionTarget<D>; COLUMNS],
    pub public_inputs: &'a [ExtensionTarget<D>; PUBLIC_INPUTS],
    pub challenges: &'a [ExtensionTarget<D>; CHALLENGES],
}
