use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use crate::plonky2::stark::config::CurtaConfig;
use crate::plonky2::stark::proof::{StarkProof, StarkProofTarget};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteStarkProof<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize> {
    main_proof: StarkProof<F, C, D>,
    lookup_proof: StarkProof<F, C, D>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteStarkProofTarget<const D: usize> {
    main_proof: StarkProofTarget<D>,
    lookup_proof: StarkProofTarget<D>,
}
