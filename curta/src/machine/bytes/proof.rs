use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;

use crate::plonky2::stark::config::CurtaConfig;
use crate::plonky2::stark::proof::{
    AirProof, AirProofTarget, StarkProofChallenges, StarkProofChallengesTarget,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteStarkProof<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize> {
    pub(crate) main_proof: AirProof<F, C, D>,
    pub(crate) lookup_proof: AirProof<F, C, D>,
    pub(crate) global_values: Vec<F>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteStarkProofTarget<const D: usize> {
    pub(crate) main_proof: AirProofTarget<D>,
    pub(crate) lookup_proof: AirProofTarget<D>,
    pub(crate) global_values: Vec<Target>,
}

pub struct ByteStarkChallenges<F: RichField + Extendable<D>, const D: usize> {
    pub(crate) main_challenges: StarkProofChallenges<F, D>,
    pub(crate) lookup_challenges: StarkProofChallenges<F, D>,
}

pub struct ByteStarkChallengesTarget<const D: usize> {
    pub(crate) main_challenges: StarkProofChallengesTarget<D>,
    pub(crate) lookup_challenges: StarkProofChallengesTarget<D>,
}
