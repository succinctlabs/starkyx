//! Prover

use core::fmt::Debug;
use core::iter::once;

use anyhow::{ensure, Result};
use plonky2::field::extension::Extendable;
use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::Field;
use plonky2::field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger::Challenger;
use plonky2::util::timing::TimingTree;
use plonky2::util::{log2_ceil, transpose};

use super::config::{CurtaConfig, StarkyConfig};
use super::Starky;
use crate::maybe_rayon::*;
use crate::plonky2::parser::consumer::ConstraintConsumer;
use crate::plonky2::parser::StarkParser;
use crate::plonky2::stark::proof::{AirProof, StarkOpeningSet, StarkProof};
use crate::plonky2::StarkyAir;
use crate::trace::generator::TraceGenerator;

#[derive(Debug, Clone)]
pub struct StarkyProver<F, C, const D: usize>(core::marker::PhantomData<(F, C)>);

#[derive(Debug)]
pub struct AirCommitment<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F>, const D: usize> {
    pub trace_commitments: Vec<PolynomialBatch<F, C::GenericConfig, D>>,
    pub public_inputs: Vec<F>,
    pub global_values: Vec<F>,
    pub challenges: Vec<F>,
}

type P<F> = <F as Packable>::Packing;

impl<F, C, const D: usize> StarkyProver<F, C, D>
where
    F: RichField + Extendable<D>,
    C: CurtaConfig<D, F = F>,
{
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    pub fn generate_trace<A, T>(
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        public_inputs: &[F],
        trace_generator: &T,
        challenger: &mut Challenger<F, C::Hasher>,
        timing: &mut TimingTree,
    ) -> Result<AirCommitment<F, C, D>>
    where
        A: StarkyAir<F, D>,
        T: TraceGenerator<F, A>,
        T::Error: Into<anyhow::Error>,
    {
        let mut challenges = vec![];
        let mut global_values = vec![F::ZERO; stark.air().num_global_values()];

        // Oberve public inputs
        challenger.observe_elements(public_inputs);

        let rate_bits = config.fri_config.rate_bits;
        let cap_height = config.fri_config.cap_height;

        let mut trace_commitments = Vec::new();
        for (r, round) in stark.air().round_data().iter().enumerate() {
            let (id_0, id_1) = round.global_values_range;
            let round_trace = trace_generator
                .generate_round(
                    stark.air(),
                    r,
                    &challenges,
                    &mut global_values[..id_1],
                    public_inputs,
                )
                .map_err(|e| e.into())?;

            let trace_cols = round_trace
                .as_columns()
                .into_par_iter()
                .map(PolynomialValues::from)
                .collect::<Vec<_>>();

            let commitment = PolynomialBatch::<F, C::GenericConfig, D>::from_values(
                trace_cols, rate_bits, false, cap_height, timing, None,
            );
            challenger.observe_elements(&global_values[id_0..id_1]);
            let cap = commitment.merkle_tree.cap.clone();
            challenger.observe_cap(&cap);
            trace_commitments.push(commitment);

            // Get the challenges for next round
            let round_challenges = challenger.get_n_challenges(round.num_challenges);
            challenges.extend(round_challenges);
        }

        Ok(AirCommitment {
            trace_commitments,
            public_inputs: public_inputs.to_vec(),
            global_values,
            challenges,
        })
    }

    pub fn prove_with_trace<A: StarkyAir<F, D>>(
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        air_commitment: AirCommitment<F, C, D>,
        challenger: &mut Challenger<F, C::Hasher>,
        timing: &mut TimingTree,
    ) -> Result<StarkProof<F, C, D>> {
        let AirCommitment {
            trace_commitments,
            public_inputs,
            global_values,
            challenges,
        } = air_commitment;
        let rate_bits = config.fri_config.rate_bits;
        let cap_height = config.fri_config.cap_height;
        let degree = 1 << trace_commitments[0].degree_log;
        let degree_bits = config.degree_bits;
        let fri_params = config.fri_params();
        assert!(
            fri_params.total_arities() <= degree_bits + rate_bits - cap_height,
            "FRI total reduction arity is too large.",
        );

        let challenge_vars = challenges
            .iter()
            .map(|x| P::<F>::from(*x))
            .collect::<Vec<_>>();
        let global_vars = global_values
            .iter()
            .map(|x| P::<F>::from(*x))
            .collect::<Vec<_>>();
        let public_vars = public_inputs
            .iter()
            .map(|x| P::<F>::from(*x))
            .collect::<Vec<_>>();
        let quotient_polys = Self::quotient_polys(
            degree_bits,
            config,
            stark,
            &trace_commitments,
            &challenge_vars,
            &global_vars,
            &public_vars,
            challenger,
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

        let quotient_commitment = PolynomialBatch::<F, C::GenericConfig, D>::from_coeffs(
            all_quotient_chunks,
            rate_bits,
            false,
            config.fri_config.cap_height,
            timing,
            None,
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
            .collect::<Vec<_>>();

        let opening_proof = PolynomialBatch::prove_openings(
            &stark.fri_instance(zeta, g, config),
            &initial_merkle_trees,
            challenger,
            &fri_params,
            timing,
        );

        let trace_caps = trace_commitments
            .into_iter()
            .map(|c| c.merkle_tree.cap)
            .collect::<Vec<_>>();
        ensure!(
            trace_caps.len() == stark.air().round_data().len(),
            "Number of trace commitments does not match"
        );
        Ok(StarkProof {
            air_proof: AirProof {
                trace_caps,
                quotient_polys_cap,
                openings,
                opening_proof,
            },
            global_values,
        })
    }

    pub fn prove<A, T>(
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        trace_generator: &T,
        public_inputs: &[F],
    ) -> Result<StarkProof<F, C, D>>
    where
        A: StarkyAir<F, D>,
        T: TraceGenerator<F, A>,
        T::Error: Into<anyhow::Error>,
    {
        let mut challenger = Challenger::<F, C::Hasher>::new();
        let mut timing = TimingTree::default();
        let air_commitment = Self::generate_trace(
            config,
            stark,
            public_inputs,
            trace_generator,
            &mut challenger,
            &mut timing,
        )?;

        Self::prove_with_trace(config, stark, air_commitment, &mut challenger, &mut timing)
    }

    #[allow(clippy::too_many_arguments)]
    fn quotient_polys<A>(
        degree_bits: usize,
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        trace_data: &[PolynomialBatch<F, C::GenericConfig, D>],
        challenges_vars: &[P<F>],
        global_vars: &[P<F>],
        public_vars: &[P<F>],
        challenger: &mut Challenger<F, C::Hasher>,
    ) -> Vec<PolynomialCoeffs<F>>
    where
        A: StarkyAir<F, D>,
    {
        let alphas = challenger.get_n_challenges(config.num_challenges);
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
        let get_trace_values_packed = |i_start| -> Vec<P<F>> {
            trace_data
                .iter()
                .flat_map(|commitment| commitment.get_lde_values_packed(i_start, step))
                .collect()
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
            .step_by(P::<F>::WIDTH)
            .flat_map_iter(|i_start| {
                let i_next_start = (i_start + next_step) % size;
                let i_range = i_start..i_start + P::<F>::WIDTH;

                let x = *P::<F>::from_slice(&coset[i_range.clone()]);
                let z_last = x - last;
                let lagrange_basis_first =
                    *P::<F>::from_slice(&lagrange_first.values[i_range.clone()]);
                let lagrange_basis_last = *P::<F>::from_slice(&lagrange_last.values[i_range]);

                let mut consumer = ConstraintConsumer::new(
                    alphas.clone(),
                    z_last,
                    lagrange_basis_first,
                    lagrange_basis_last,
                );
                let mut parser = StarkParser {
                    local_vars: &get_trace_values_packed(i_start),
                    next_vars: &get_trace_values_packed(i_next_start),
                    global_vars,
                    public_vars,
                    challenges: challenges_vars,
                    consumer: &mut consumer,
                };

                stark.air().eval(&mut parser);

                let mut constraints_evals = consumer.accumulators();
                // We divide the constraints evaluations by `Z_H(x)`.
                let denominator_inv: P<F> = z_h_on_coset.eval_inverse_packed(i_start);

                for eval in &mut constraints_evals {
                    *eval *= denominator_inv;
                }

                let num_challenges = alphas.len();

                (0..P::<F>::WIDTH).map(move |i| {
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
