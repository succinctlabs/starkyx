//! Prover

use core::iter::once;

use anyhow::{self, ensure, Result};
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::Field;
use plonky2::field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::config::GenericConfig;
use plonky2::util::timing::TimingTree;
use plonky2::util::{log2_ceil, transpose};

use super::config::StarkyConfig;
use super::Plonky2Stark;
use crate::air::RAir;
use crate::maybe_rayon::*;
use crate::plonky2::challenger::Plonky2Challenger;
use crate::plonky2::parser::consumer::ConstraintConsumer;
use crate::plonky2::parser::StarkParser;
use crate::plonky2::stark::proof::{StarkOpeningSet, StarkProof};
use crate::trace::generator::TraceGenerator;

#[derive(Debug, Clone)]
pub struct StarkyProver<F, C, FE, P, const D: usize, const D2: usize>(
    core::marker::PhantomData<(F, C, FE, P)>,
);

impl<F, C, P, const D: usize> StarkyProver<F, C, F, P, D, 1>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    P: PackedField<Scalar = F>,
{
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    pub fn prove<S, T>(
        config: &StarkyConfig<F, C, D>,
        stark: &S,
        trace_generator: &T,
        public_inputs: &[F],
    ) -> Result<StarkProof<F, C, D>>
    where
        S: Plonky2Stark<F, D>,
        S::Air: for<'a> RAir<StarkParser<'a, F, F, P, D, 1>>,
        T: TraceGenerator<F, S::Air>,
        T::Error: Into<anyhow::Error>,
        [(); S::COLUMNS]:,
    {
        let mut challenger = Plonky2Challenger::<F, C::Hasher>::new();

        // Obsetrve public inputs
        challenger.0.observe_elements(public_inputs);

        let rate_bits = config.fri_config.rate_bits;
        let cap_height = config.fri_config.cap_height;

        let mut challenges = vec![];

        let mut trace_commitments = Vec::new();
        let mut timing = TimingTree::default();
        let mut num_rows = 0;
        for r in 0..stark.air().num_rounds() {
            let round_trace = trace_generator
                .generate_round(stark.air(), r, &challenges, public_inputs)
                .map_err(|e| e.into())?;

            let trace_cols = round_trace
                .as_columns()
                .into_par_iter()
                .map(PolynomialValues::from)
                .collect::<Vec<_>>();

            if r == 0 {
                num_rows = trace_cols[0].len();
            }

            let commitment = PolynomialBatch::<F, C, D>::from_values(
                trace_cols,
                rate_bits,
                false,
                cap_height,
                &mut timing,
                None,
            );
            let cap = commitment.merkle_tree.cap.clone();
            challenger.0.observe_cap(&cap);
            trace_commitments.push(commitment);

            // Get the challenges for next round
            let round_challenges = challenger.0.get_n_challenges(stark.air().num_challenges(r));
            challenges.extend(round_challenges);
        }

        let degree = num_rows;
        let degree_bits = config.degree_bits;
        let fri_params = config.fri_params();
        assert!(
            fri_params.total_arities() <= degree_bits + rate_bits - cap_height,
            "FRI total reduction arity is too large.",
        );

        let challenge_vars = challenges.into_iter().map(P::from).collect::<Vec<_>>();
        let public_input_vars = public_inputs
            .iter()
            .map(|x| P::from(*x))
            .collect::<Vec<_>>();
        let quotient_polys = Self::quotient_polys(
            degree_bits,
            config,
            stark,
            &trace_commitments,
            &challenge_vars,
            &public_input_vars,
            &mut challenger,
        );
        let quotient_degree_factor = stark.air().quotient_degree_factor();
        let all_quotient_chunks = quotient_polys
            .into_par_iter()
            .flat_map(|mut quotient_poly| {
                quotient_poly
                    .trim_to_len(degree * quotient_degree_factor)
                    .expect(
                        "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                    );
                // Split quotient into degree-n chunks.
                quotient_poly.chunks(degree)
            })
            .collect();

        let quotient_commitment = PolynomialBatch::<F, C, D>::from_coeffs(
            all_quotient_chunks,
            rate_bits,
            false,
            config.fri_config.cap_height,
            &mut timing,
            None,
        );

        let quotient_polys_cap = quotient_commitment.merkle_tree.cap.clone();
        challenger.0.observe_cap(&quotient_polys_cap);

        let zeta = challenger.0.get_extension_challenge::<D>();
        // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
        // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
        // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
        let g = F::primitive_root_of_unity(degree_bits);
        ensure!(
            zeta.exp_power_of_2(degree_bits) != F::Extension::ONE,
            "Opening point is in the subgroup."
        );
        let openings = StarkOpeningSet::new(zeta, g, &trace_commitments, &quotient_commitment);
        challenger.0.observe_openings(&openings.to_fri_openings());

        let initial_merkle_trees = trace_commitments
            .iter()
            .chain(once(&quotient_commitment))
            .collect::<Vec<_>>();

        let opening_proof = PolynomialBatch::prove_openings(
            &stark.fri_instance(zeta, g, config),
            &initial_merkle_trees,
            &mut challenger.0,
            &fri_params,
            &mut timing,
        );

        let trace_caps = trace_commitments
            .into_iter()
            .map(|c| c.merkle_tree.cap)
            .collect::<Vec<_>>();
        ensure!(
            trace_caps.len() == stark.air().round_lengths().len(),
            "Number of trace commitments does not match"
        );
        Ok(StarkProof {
            trace_caps,
            quotient_polys_cap,
            openings,
            opening_proof,
        })
    }

    fn quotient_polys<S>(
        degree_bits: usize,
        config: &StarkyConfig<F, C, D>,
        stark: &S,
        trace_data: &[PolynomialBatch<F, C, D>],
        challenges_vars: &[P],
        public_inputs_vars: &[P],
        challenger: &mut Plonky2Challenger<F, C::Hasher>,
    ) -> Vec<PolynomialCoeffs<F>>
    where
        S: Plonky2Stark<F, D>,
        S::Air: for<'a> RAir<StarkParser<'a, F, F, P, D, 1>>,
        [(); S::COLUMNS]:,
    {
        let alphas = challenger.0.get_n_challenges(config.num_challenges);
        let degree = 1 << degree_bits;
        let rate_bits = config.fri_config.rate_bits;

        let quotient_degree_bits = log2_ceil(stark.air().quotient_degree_factor());
        assert!(
            quotient_degree_bits <= rate_bits,
            "Having constraints of degree higher than the rate is not supported yet."
        );
        let step = 1 << (rate_bits - quotient_degree_bits);
        // When opening the `Z`s polys at the "next" point, need to look at the point `next_step` steps away.
        let next_step = 1 << quotient_degree_bits;

        // Evaluation of the first Lagrange polynomial on the LDE domain.
        let lagrange_first =
            PolynomialValues::selector(degree, 0).lde_onto_coset(quotient_degree_bits);
        // Evaluation of the last Lagrange polynomial on the LDE domain.
        let lagrange_last =
            PolynomialValues::selector(degree, degree - 1).lde_onto_coset(quotient_degree_bits);

        let z_h_on_coset = ZeroPolyOnCoset::<F>::new(degree_bits, quotient_degree_bits);

        // Retrieve the LDE values at index `i`.
        let get_trace_values_packed = |i_start| -> [P; S::COLUMNS] {
            let trace = trace_data
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
                let mut parser = StarkParser {
                    local_vars: &get_trace_values_packed(i_start),
                    next_vars: &get_trace_values_packed(i_next_start),
                    public_inputs: public_inputs_vars,
                    challenges: challenges_vars,
                    consumer: &mut consumer,
                };

                stark.air().eval(&mut parser);

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
}
