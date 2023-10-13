use plonky2::field::extension::Extendable;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::RichField;

use crate::plonky2::stark::config::CurtaConfig;
use crate::plonky2::stark::proof::{AirProof, StarkProof, StarkProofTarget};
use crate::plonky2::stark::prover::AirCommitment;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteStarkProof<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize> {
    pub(crate) main_proof: AirProof<F, C, D>,
    pub(crate) lookup_proof: AirProof<F, C, D>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteStarkProofTarget<const D: usize> {
    pub(crate) main_proof: StarkProofTarget<D>,
    pub(crate) lookup_proof: StarkProofTarget<D>,
}

#[derive(Debug)]
pub struct ByteAirCommitment<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize>
{
    pub(crate) main_trace_commitments: Vec<PolynomialBatch<F, C::GenericConfig, D>>,
    pub(crate) lookup_trace_commitments: Vec<PolynomialBatch<F, C::GenericConfig, D>>,
    pub(crate) public_inputs: Vec<F>,
    pub(crate) global_values: Vec<F>,
    pub(crate) challenges: Vec<F>,
}
