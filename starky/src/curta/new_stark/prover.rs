use alloc::vec::Vec;
use core::iter::once;

use anyhow::{ensure, Result};
use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::Field;
use plonky2::field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger::Challenger;
use plonky2::plonk::config::{GenericConfig, Hasher};
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use plonky2::util::{log2_ceil, log2_strict, transpose};
use plonky2_maybe_rayon::*;

use super::proof::{StarkOpeningSet, StarkProof, StarkProofWithPublicInputs};
use super::vanishing_poly::eval_vanishing_poly;
use super::vars::StarkEvaluationVars;
use super::Stark;
use crate::config::StarkConfig;
use crate::constraint_consumer::ConstraintConsumer;
use crate::curta::trace::types::StarkTraceGenerator;

pub fn prove<F, C, S, T, const D: usize, const R: usize>(
    stark: S,
    config: &StarkConfig,
    mut witness_generator: T,
    num_rows: usize,
    public_inputs: [F; S::PUBLIC_INPUTS],
    timing: &mut TimingTree,
) -> Result<StarkProofWithPublicInputs<F, C, D, R>>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D, R>,
    T: StarkTraceGenerator<S, F, D, R>,
    [(); S::COLUMNS]:,
    [(); S::PUBLIC_INPUTS]:,
    [(); S::CHALLENGES]:,
    [(); C::Hasher::HASH_SIZE]:,
{
    let mut challenger: Challenger<F, <C as GenericConfig<D>>::Hasher> = Challenger::new();
    let rate_bits = config.fri_config.rate_bits;
    let cap_height = config.fri_config.cap_height;

    let mut trace_commitments = Vec::new();
    let mut challenges = Vec::with_capacity(S::CHALLENGES);

    for r in 0..R {
        let trace_values = timed!(
            timing,
            &format!("Generate trace for round {}", r),
            witness_generator.generate_round(&stark, r, &challenges)
        );
        let commitment = timed!(
            timing,
            &format!("compute trace commitment for round {}", r),
            PolynomialBatch::<F, C, D>::from_values(
                trace_values,
                rate_bits,
                false,
                cap_height,
                timing,
                None,
            )
        );
        let cap = commitment.merkle_tree.cap.clone();
        challenger.observe_cap(&cap);
        trace_commitments.push(commitment);

        // Get the challenges for next round
        let round_challenges = challenger.get_n_challenges(stark.num_challenges(r));
        challenges.extend(round_challenges);
    }

    let degree = num_rows;
    let degree_bits = log2_strict(degree);
    let fri_params = config.fri_params(degree_bits);
    assert!(
        fri_params.total_arities() <= degree_bits + rate_bits - cap_height,
        "FRI total reduction arity is too large.",
    );

    let alphas = challenger.get_n_challenges(config.num_challenges);
    ensure!(
        challenges.len() == S::CHALLENGES,
        "Number of challenges does not match"
    );
    let challenges_array: [F; S::CHALLENGES] = challenges.try_into().unwrap();
    let quotient_polys = compute_quotient_polys::<F, <F as Packable>::Packing, C, S, D, R>(
        &stark,
        &trace_commitments,
        public_inputs,
        challenges_array,
        alphas,
        degree_bits,
        config,
    );
    let all_quotient_chunks = quotient_polys
        .into_par_iter()
        .flat_map(|mut quotient_poly| {
            quotient_poly
                .trim_to_len(degree * stark.quotient_degree_factor())
                .expect("Quotient has failed, the vanishing polynomial is not divisible by Z_H");
            // Split quotient into degree-n chunks.
            quotient_poly.chunks(degree)
        })
        .collect();
    let quotient_commitment = timed!(
        timing,
        "compute quotient commitment",
        PolynomialBatch::from_coeffs(
            all_quotient_chunks,
            rate_bits,
            false,
            config.fri_config.cap_height,
            timing,
            None,
        )
    );
    let quotient_polys_cap = quotient_commitment.merkle_tree.cap.clone();
    challenger.observe_cap(&quotient_polys_cap);

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
    // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
    let g = F::primitive_root_of_unity(degree_bits);
    ensure!(
        zeta.exp_power_of_2(degree_bits) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );
    let openings = StarkOpeningSet::new(zeta, g, &trace_commitments, &quotient_commitment);
    challenger.observe_openings(&openings.to_fri_openings());

    let initial_merkle_trees = trace_commitments
        .iter()
        .chain(once(&quotient_commitment))
        .collect_vec();

    let opening_proof = timed!(
        timing,
        "compute openings proof",
        PolynomialBatch::prove_openings(
            &stark.fri_instance(zeta, g, config),
            &initial_merkle_trees,
            &mut challenger,
            &fri_params,
            timing,
        )
    );

    let trace_caps = trace_commitments
        .into_iter()
        .map(|c| c.merkle_tree.cap)
        .collect_vec();
    ensure!(
        trace_caps.len() == R,
        "Number of trace commitments does not match"
    );
    let trace_caps: [_; R] = trace_caps.try_into().unwrap();
    let proof = StarkProof {
        trace_caps,
        quotient_polys_cap,
        openings,
        opening_proof,
    };

    Ok(StarkProofWithPublicInputs {
        proof,
        public_inputs: public_inputs.to_vec(),
    })
}

/// Computes the quotient polynomials `(sum alpha^i C_i(x)) / Z_H(x)` for `alpha` in `alphas`,
/// where the `C_i`s are the Stark constraints.
fn compute_quotient_polys<'a, F, P, C, S, const D: usize, const R: usize>(
    stark: &S,
    trace_commitment: &'a [PolynomialBatch<F, C, D>],
    public_inputs: [F; S::PUBLIC_INPUTS],
    challenges: [F; S::CHALLENGES],
    alphas: Vec<F>,
    degree_bits: usize,
    config: &StarkConfig,
) -> Vec<PolynomialCoeffs<F>>
where
    F: RichField + Extendable<D>,
    P: PackedField<Scalar = F>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D, R>,
    [(); S::COLUMNS]:,
    [(); S::PUBLIC_INPUTS]:,
    [(); S::CHALLENGES]:,
{
    let degree = 1 << degree_bits;
    let rate_bits = config.fri_config.rate_bits;

    let quotient_degree_bits = log2_ceil(stark.quotient_degree_factor());
    assert!(
        quotient_degree_bits <= rate_bits,
        "Having constraints of degree higher than the rate is not supported yet."
    );
    let step = 1 << (rate_bits - quotient_degree_bits);
    // When opening the `Z`s polys at the "next" point, need to look at the point `next_step` steps away.
    let next_step = 1 << quotient_degree_bits;

    // Evaluation of the first Lagrange polynomial on the LDE domain.
    let lagrange_first = PolynomialValues::selector(degree, 0).lde_onto_coset(quotient_degree_bits);
    // Evaluation of the last Lagrange polynomial on the LDE domain.
    let lagrange_last =
        PolynomialValues::selector(degree, degree - 1).lde_onto_coset(quotient_degree_bits);

    let z_h_on_coset = ZeroPolyOnCoset::<F>::new(degree_bits, quotient_degree_bits);

    // Retrieve the LDE values at index `i`.
    let get_trace_values_packed = |i_start| -> [P; S::COLUMNS] {
        let trace = trace_commitment
            .iter()
            .flat_map(|commitment| commitment.get_lde_values_packed(i_start, step))
            .collect::<Vec<_>>();
        // .try_into()
        // .expect("Invalid number of trace columns")
        assert_eq!(trace.len(), S::COLUMNS);
        trace.try_into().unwrap()
    };
    // Last element of the subgroup.
    let last = F::primitive_root_of_unity(degree_bits).inverse();
    let size = degree << quotient_degree_bits;
    let coset = F::cyclic_subgroup_coset_known_order(
        F::primitive_root_of_unity(degree_bits + quotient_degree_bits),
        F::coset_shift(),
        size,
    );

    // We will step by `P::WIDTH`, and in each iteration, evaluate the quotient polynomial at
    // a batch of `P::WIDTH` points.
    let quotient_values = (0..size)
        .into_par_iter()
        .step_by(P::WIDTH)
        .flat_map_iter(|i_start| {
            let i_next_start = (i_start + next_step) % size;
            let i_range = i_start..i_start + P::WIDTH;

            let x = *P::from_slice(&coset[i_range.clone()]);
            let z_last = x - last;
            let lagrange_basis_first = *P::from_slice(&lagrange_first.values[i_range.clone()]);
            let lagrange_basis_last = *P::from_slice(&lagrange_last.values[i_range]);

            let mut consumer = ConstraintConsumer::new(
                alphas.clone(),
                z_last,
                lagrange_basis_first,
                lagrange_basis_last,
            );
            let vars = StarkEvaluationVars {
                local_values: &get_trace_values_packed(i_start),
                next_values: &get_trace_values_packed(i_next_start),
                public_inputs: &public_inputs,
                challenges: &challenges,
            };
            eval_vanishing_poly::<F, F, P, S, D, 1, R>(&stark, vars, &mut consumer);

            let mut constraints_evals = consumer.accumulators();
            // We divide the constraints evaluations by `Z_H(x)`.
            let denominator_inv: P = z_h_on_coset.eval_inverse_packed(i_start);

            for eval in &mut constraints_evals {
                *eval *= denominator_inv;
            }

            let num_challenges = alphas.len();

            (0..P::WIDTH).map(move |i| {
                (0..num_challenges)
                    .map(|j| constraints_evals[j].as_slice()[i])
                    .collect()
            })
        })
        .collect::<Vec<_>>();
    transpose(&quotient_values)
        .into_par_iter()
        .map(PolynomialValues::new)
        .map(|values| values.coset_ifft(F::coset_shift()))
        .collect()
}
