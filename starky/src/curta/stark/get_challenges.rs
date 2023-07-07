use alloc::vec::Vec;

use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialCoeffs;
use plonky2::fri::proof::{FriProof, FriProofTarget};
use plonky2::gadgets::polynomial::PolynomialCoeffsExtTarget;
use plonky2::hash::hash_types::{MerkleCapTarget, RichField};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::challenger::{Challenger, RecursiveChallenger};
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

use super::proof::*;
use super::Stark;
use crate::config::StarkConfig;

fn get_challenges<F, C, S, const D: usize, const R: usize>(
    stark: &S,
    trace_caps: &[MerkleCap<F, C::Hasher>],
    quotient_polys_cap: &MerkleCap<F, C::Hasher>,
    openings: &StarkOpeningSet<F, D>,
    commit_phase_merkle_caps: &[MerkleCap<F, C::Hasher>],
    final_poly: &PolynomialCoeffs<F::Extension>,
    pow_witness: F,
    config: &StarkConfig,
    degree_bits: usize,
) -> StarkProofChallenges<F, D, R>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D, R>,
{
    let num_challenges = config.num_challenges;

    let mut challenger = Challenger::<F, C::Hasher>::new();

    let mut challenges = Vec::with_capacity(S::CHALLENGES);

    for (r, cap) in trace_caps.iter().enumerate() {
        challenger.observe_cap(cap);
        let round_challenges = challenger.get_n_challenges(stark.num_challenges(r));
        challenges.extend(round_challenges);
    }

    let stark_alphas = challenger.get_n_challenges(num_challenges);

    challenger.observe_cap(quotient_polys_cap);
    let stark_zeta = challenger.get_extension_challenge::<D>();

    challenger.observe_openings(&openings.to_fri_openings());

    StarkProofChallenges {
        stark_alphas,
        stark_betas: challenges,
        stark_zeta,
        fri_challenges: challenger.fri_challenges::<C, D>(
            commit_phase_merkle_caps,
            final_poly,
            pow_witness,
            degree_bits,
            &config.fri_config,
        ),
    }
}

impl<F, C, const D: usize, const R: usize> StarkProofWithPublicInputs<F, C, D, R>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    // TODO: Should be used later in compression?
    #![allow(dead_code)]
    pub(crate) fn fri_query_indices<S: Stark<F, D, R>>(
        &self,
        stark: &S,
        config: &StarkConfig,
        degree_bits: usize,
    ) -> Vec<usize> {
        self.get_challenges(stark, config, degree_bits)
            .fri_challenges
            .fri_query_indices
    }

    /// Computes all Fiat-Shamir challenges used in the STARK proof.
    pub(crate) fn get_challenges<S: Stark<F, D, R>>(
        &self,
        stark: &S,
        config: &StarkConfig,
        degree_bits: usize,
    ) -> StarkProofChallenges<F, D, R> {
        let StarkProof {
            trace_caps,
            quotient_polys_cap,
            openings,
            opening_proof:
                FriProof {
                    commit_phase_merkle_caps,
                    final_poly,
                    pow_witness,
                    ..
                },
        } = &self.proof;

        get_challenges::<F, C, S, D, R>(
            stark,
            trace_caps,
            quotient_polys_cap,
            openings,
            commit_phase_merkle_caps,
            final_poly,
            *pow_witness,
            config,
            degree_bits,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_challenges_target<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D, R>,
    const D: usize,
    const R: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &S,
    trace_caps: &[MerkleCapTarget],
    quotient_polys_cap: &MerkleCapTarget,
    openings: &StarkOpeningSetTarget<D>,
    commit_phase_merkle_caps: &[MerkleCapTarget],
    final_poly: &PolynomialCoeffsExtTarget<D>,
    pow_witness: Target,
    config: &StarkConfig,
) -> StarkProofChallengesTarget<D>
where
    C::Hasher: AlgebraicHasher<F>,
{
    let num_challenges = config.num_challenges;

    let mut challenger = RecursiveChallenger::<F, C::Hasher, D>::new(builder);

    let mut challenges = Vec::with_capacity(S::CHALLENGES);

    for (r, cap) in trace_caps.iter().enumerate() {
        challenger.observe_cap(cap);
        let round_challenges = challenger.get_n_challenges(builder, stark.num_challenges(r));
        challenges.extend(round_challenges);
    }
    let stark_alphas = challenger.get_n_challenges(builder, num_challenges);

    challenger.observe_cap(quotient_polys_cap);
    let stark_zeta = challenger.get_extension_challenge(builder);

    challenger.observe_openings(&openings.to_fri_openings());

    StarkProofChallengesTarget {
        stark_alphas,
        stark_betas: challenges,
        stark_zeta,
        fri_challenges: challenger.fri_challenges(
            builder,
            commit_phase_merkle_caps,
            final_poly,
            pow_witness,
            &config.fri_config,
        ),
    }
}

impl<const D: usize, const R: usize> StarkProofWithPublicInputsTarget<D, R> {
    pub(crate) fn get_challenges<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        S: Stark<F, D, R>,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        stark: &S,
        config: &StarkConfig,
    ) -> StarkProofChallengesTarget<D>
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        let StarkProofTarget {
            trace_caps,
            quotient_polys_cap,
            openings,
            opening_proof:
                FriProofTarget {
                    commit_phase_merkle_caps,
                    final_poly,
                    pow_witness,
                    ..
                },
        } = &self.proof;

        get_challenges_target::<F, C, S, D, R>(
            builder,
            stark,
            trace_caps,
            quotient_polys_cap,
            openings,
            commit_phase_merkle_caps,
            final_poly,
            *pow_witness,
            config,
        )
    }
}