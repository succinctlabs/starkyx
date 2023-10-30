use itertools::Itertools;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::fri::proof::{FriChallenges, FriChallengesTarget, FriProof, FriProofTarget};
use plonky2::fri::structure::{
    FriOpeningBatch, FriOpeningBatchTarget, FriOpenings, FriOpeningsTarget,
};
use plonky2::hash::hash_types::{MerkleCapTarget, RichField};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::challenger::{Challenger, RecursiveChallenger};
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::GenericConfig;
use serde::{Deserialize, Serialize};

use super::config::{CurtaConfig, StarkyConfig};
use super::Starky;
use crate::air::{RAir, RAirData};
use crate::maybe_rayon::*;
use crate::plonky2::parser::RecursiveStarkParser;
use crate::utils::serde::{
    deserialize_extension_targets, deserialize_fri_proof_target, deserialize_merkle_cap_target,
    deserialize_merkle_cap_targets, serialize_extension_targets, serialize_fri_proof_target,
    serialize_merkle_cap_target, serialize_merkle_cap_targets,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AirProof<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize> {
    /// Merkle cap of LDEs of trace values for each round.
    pub trace_caps: Vec<MerkleCap<F, C::Hasher>>,
    /// Merkle cap of LDEs of trace values.
    pub quotient_polys_cap: MerkleCap<F, C::Hasher>,
    /// Purported values of each polynomial at the challenge point.
    pub openings: StarkOpeningSet<F, D>,
    /// A batch FRI argument for all openings.
    pub opening_proof: FriProof<F, C::Hasher, D>,
}

impl<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize> AirProof<F, C, D> {
    pub fn get_iop_challenges(
        &self,
        config: &StarkyConfig<C, D>,
        degree_bits: usize,
        air_challenges: Vec<F>,
        challenger: &mut Challenger<F, C::Hasher>,
    ) -> StarkProofChallenges<F, D> {
        let AirProof {
            quotient_polys_cap,
            openings,
            opening_proof:
                FriProof {
                    commit_phase_merkle_caps,
                    final_poly,
                    pow_witness,
                    ..
                },
            ..
        } = &self;

        let num_challenges = config.num_challenges;
        let stark_alphas = challenger.get_n_challenges(num_challenges);

        challenger.observe_cap(quotient_polys_cap);
        let stark_zeta = challenger.get_extension_challenge::<D>();

        challenger.observe_openings(&openings.to_fri_openings());

        StarkProofChallenges {
            stark_alphas,
            stark_betas: air_challenges,
            stark_zeta,
            fri_challenges: challenger.fri_challenges::<C::GenericConfig, D>(
                commit_phase_merkle_caps,
                final_poly,
                *pow_witness,
                degree_bits,
                &config.fri_config,
            ),
        }
    }
}

/// A proof of a STARK computation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct StarkProof<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize> {
    pub air_proof: AirProof<F, C, D>,
    pub global_values: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize> StarkProof<F, C, D> {
    /// Recover the length of the trace from a STARK proof and a STARK config.
    pub fn recover_degree_bits(&self, config: &StarkyConfig<C, D>) -> usize {
        let initial_merkle_proof = &self.air_proof.opening_proof.query_round_proofs[0]
            .initial_trees_proof
            .evals_proofs[0]
            .1;
        let lde_bits = config.fri_config.cap_height + initial_merkle_proof.siblings.len();
        lde_bits - config.fri_config.rate_bits
    }

    pub fn get_iop_challenges(
        &self,
        config: &StarkyConfig<C, D>,
        degree_bits: usize,
        air_challenges: Vec<F>,
        challenger: &mut Challenger<F, C::Hasher>,
    ) -> StarkProofChallenges<F, D> {
        self.air_proof
            .get_iop_challenges(config, degree_bits, air_challenges, challenger)
    }

    pub(crate) fn get_challenges<A: RAirData>(
        &self,
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        public_inputs: &[F],
        degree_bits: usize,
    ) -> StarkProofChallenges<F, D> {
        let StarkProof {
            air_proof: AirProof { trace_caps, .. },
            global_values,
            ..
        } = &self;

        let mut challenger = Challenger::<F, C::Hasher>::new();
        // Observe public inputs
        challenger.observe_elements(public_inputs);

        let mut challenges = vec![];
        for (round, cap) in stark.air().round_data().iter().zip_eq(trace_caps.iter()) {
            let (id_0, id_1) = round.global_values_range;
            challenger.observe_elements(&global_values[id_0..id_1]);
            challenger.observe_cap(cap);
            let round_challenges = challenger.get_n_challenges(round.num_challenges);
            challenges.extend(round_challenges);
        }

        self.get_iop_challenges(config, degree_bits, challenges, &mut challenger)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AirProofTarget<const D: usize> {
    #[serde(serialize_with = "serialize_merkle_cap_targets")]
    #[serde(deserialize_with = "deserialize_merkle_cap_targets")]
    pub trace_caps: Vec<MerkleCapTarget>,
    #[serde(serialize_with = "serialize_merkle_cap_target")]
    #[serde(deserialize_with = "deserialize_merkle_cap_target")]
    pub quotient_polys_cap: MerkleCapTarget,
    pub openings: StarkOpeningSetTarget<D>,
    #[serde(serialize_with = "serialize_fri_proof_target")]
    #[serde(deserialize_with = "deserialize_fri_proof_target")]
    pub opening_proof: FriProofTarget<D>,
}

impl<const D: usize> AirProofTarget<D> {
    pub fn get_iop_challenges_target<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        config: &StarkyConfig<C, D>,
        air_challengs: Vec<Target>,
        challenger: &mut RecursiveChallenger<F, C::InnerHasher, D>,
    ) -> StarkProofChallengesTarget<D> {
        let AirProofTarget {
            quotient_polys_cap,
            openings,
            opening_proof:
                FriProofTarget {
                    commit_phase_merkle_caps,
                    final_poly,
                    pow_witness,
                    ..
                },
            ..
        } = &self;

        let num_challenges = config.num_challenges;

        let stark_alphas = challenger.get_n_challenges(builder, num_challenges);

        challenger.observe_cap(quotient_polys_cap);
        let stark_zeta = challenger.get_extension_challenge(builder);

        challenger.observe_openings(&openings.to_fri_openings());

        StarkProofChallengesTarget {
            stark_alphas,
            stark_betas: air_challengs,
            stark_zeta,
            fri_challenges: challenger.fri_challenges(
                builder,
                commit_phase_merkle_caps,
                final_poly,
                *pow_witness,
                &config.fri_config,
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarkProofTarget<const D: usize> {
    pub air_proof: AirProofTarget<D>,
    pub global_values: Vec<Target>,
}

impl<const D: usize> StarkProofTarget<D> {
    /// Recover the length of the trace from a STARK proof and a STARK config.
    pub fn recover_degree_bits<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>>(
        &self,
        config: &StarkyConfig<C, D>,
    ) -> usize {
        let initial_merkle_proof = &self.air_proof.opening_proof.query_round_proofs[0]
            .initial_trees_proof
            .evals_proofs[0]
            .1;
        let lde_bits = config.fri_config.cap_height + initial_merkle_proof.siblings.len();
        lde_bits - config.fri_config.rate_bits
    }

    pub fn get_iop_challenges_target<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        config: &StarkyConfig<C, D>,
        air_challengs: Vec<Target>,
        challenger: &mut RecursiveChallenger<F, C::InnerHasher, D>,
    ) -> StarkProofChallengesTarget<D> {
        self.air_proof
            .get_iop_challenges_target(builder, config, air_challengs, challenger)
    }

    pub fn get_challenges_target<
        F: RichField + Extendable<D>,
        A: for<'a> RAir<RecursiveStarkParser<'a, F, D>>,
        C: CurtaConfig<D, F = F>,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        config: &StarkyConfig<C, D>,
        public_inputs: &[Target],
        stark: &Starky<A>,
    ) -> StarkProofChallengesTarget<D> {
        let StarkProofTarget {
            air_proof: AirProofTarget { trace_caps, .. },
            global_values,
            ..
        } = &self;

        let mut challenger = RecursiveChallenger::<F, C::InnerHasher, D>::new(builder);

        // Observe public inputs
        challenger.observe_elements(public_inputs);

        let mut challenges = vec![];
        for (round, cap) in stark.air().round_data().iter().zip(trace_caps.iter()) {
            let (id_0, id_1) = round.global_values_range;
            challenger.observe_elements(&global_values[id_0..id_1]);
            challenger.observe_cap(cap);
            let round_challenges = challenger.get_n_challenges(builder, round.num_challenges);
            challenges.extend(round_challenges);
        }

        self.get_iop_challenges_target(builder, config, challenges, &mut challenger)
    }
}

pub struct StarkProofChallenges<F: RichField + Extendable<D>, const D: usize> {
    /// Random values used to combine STARK constraints.
    pub stark_alphas: Vec<F>,

    /// Random values that can be used by the STARK for any purpose.
    pub stark_betas: Vec<F>,

    /// Point at which the STARK polynomials are opened.
    pub stark_zeta: F::Extension,

    pub fri_challenges: FriChallenges<F, D>,
}

pub struct StarkProofChallengesTarget<const D: usize> {
    pub stark_alphas: Vec<Target>,
    pub stark_betas: Vec<Target>,
    pub stark_zeta: ExtensionTarget<D>,
    pub fri_challenges: FriChallengesTarget<D>,
}

/// Purported values of each polynomial at the challenge point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct StarkOpeningSet<F: RichField + Extendable<D>, const D: usize> {
    pub local_values: Vec<F::Extension>,
    pub next_values: Vec<F::Extension>,
    pub quotient_polys: Vec<F::Extension>,
}

impl<F: RichField + Extendable<D>, const D: usize> StarkOpeningSet<F, D> {
    pub fn new<C: GenericConfig<D, F = F>>(
        zeta: F::Extension,
        g: F,
        trace_commitments: &[PolynomialBatch<F, C, D>],
        quotient_commitment: &PolynomialBatch<F, C, D>,
    ) -> Self {
        let eval_commitment = |z: F::Extension, c: &PolynomialBatch<F, C, D>| {
            c.polynomials
                .par_iter()
                .map(|p| p.to_extension().eval(z))
                .collect::<Vec<_>>()
        };
        let zeta_next = zeta.scalar_mul(g);

        let local_values = trace_commitments
            .par_iter()
            .flat_map(|trace| eval_commitment(zeta, trace))
            .collect::<Vec<_>>();
        let next_values = trace_commitments
            .par_iter()
            .flat_map(|trace| eval_commitment(zeta_next, trace))
            .collect::<Vec<_>>();
        let quotient_polys = eval_commitment(zeta, quotient_commitment);
        Self {
            local_values,
            next_values,
            quotient_polys,
        }
    }

    pub(crate) fn to_fri_openings(&self) -> FriOpenings<F, D> {
        let zeta_batch = FriOpeningBatch {
            values: self
                .local_values
                .iter()
                .chain(&self.quotient_polys)
                .copied()
                .collect::<Vec<_>>(),
        };
        let zeta_next_batch = FriOpeningBatch {
            values: self.next_values.to_vec(),
        };
        FriOpenings {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarkOpeningSetTarget<const D: usize> {
    #[serde(serialize_with = "serialize_extension_targets")]
    #[serde(deserialize_with = "deserialize_extension_targets")]
    pub local_values: Vec<ExtensionTarget<D>>,
    #[serde(serialize_with = "serialize_extension_targets")]
    #[serde(deserialize_with = "deserialize_extension_targets")]
    pub next_values: Vec<ExtensionTarget<D>>,
    #[serde(serialize_with = "serialize_extension_targets")]
    #[serde(deserialize_with = "deserialize_extension_targets")]
    pub quotient_polys: Vec<ExtensionTarget<D>>,
}

impl<const D: usize> StarkOpeningSetTarget<D> {
    pub(crate) fn to_fri_openings(&self) -> FriOpeningsTarget<D> {
        let zeta_batch = FriOpeningBatchTarget {
            values: self
                .local_values
                .iter()
                .chain(&self.quotient_polys)
                .copied()
                .collect::<Vec<_>>(),
        };
        let zeta_next_batch = FriOpeningBatchTarget {
            values: self.next_values.to_vec(),
        };
        FriOpeningsTarget {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}
