use plonky2::field::extension::Extendable;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;

use crate::plonky2::stark::config::CurtaConfig;
use crate::plonky2::stark::proof::{AirProof, AirProofTarget};
use crate::plonky2::stark::prover::AirCommitment;

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

#[derive(Debug)]
pub struct ByteAirCommitment<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize>
{
    pub(crate) main_trace_commitments: Vec<PolynomialBatch<F, C::GenericConfig, D>>,
    pub(crate) lookup_trace_commitments: Vec<PolynomialBatch<F, C::GenericConfig, D>>,
    pub(crate) public_inputs: Vec<F>,
    pub(crate) global_values: Vec<F>,
    pub(crate) challenges: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize>
    ByteAirCommitment<F, C, D>
{
    pub fn air_commitments(self) -> (AirCommitment<F, C, D>, AirCommitment<F, C, D>) {
        let ByteAirCommitment {
            main_trace_commitments,
            lookup_trace_commitments,
            public_inputs,
            global_values,
            challenges,
        } = self;

        let main_commitment = AirCommitment {
            trace_commitments: main_trace_commitments,
            public_inputs: public_inputs.clone(),
            global_values: global_values.clone(),
            challenges: challenges.clone(),
        };

        let lookup_commitment = AirCommitment {
            trace_commitments: lookup_trace_commitments,
            public_inputs,
            global_values,
            challenges,
        };

        (main_commitment, lookup_commitment)
    }
}
