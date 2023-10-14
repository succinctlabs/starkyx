use core::iter::once;
use std::collections::HashMap;

use anyhow::{ensure, Result};
use itertools::Itertools;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::types::Field as Plonky2Field;
use plonky2::fri::verifier::verify_fri_proof;
use plonky2::fri::witness_util::set_fri_proof_target;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::iop::witness::WitnessWrite;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::AlgebraicHasher;
use plonky2::plonk::plonk_common::reduce_with_powers;
use plonky2::util::reducing::ReducingFactorTarget;

use super::config::{CurtaConfig, StarkyConfig};
use super::proof::{
    AirProofTarget, StarkOpeningSet, StarkOpeningSetTarget, StarkProof, StarkProofChallenges,
    StarkProofChallengesTarget, StarkProofTarget,
};
use super::Starky;
use crate::air::{RAir, RAirData};
use crate::plonky2::parser::consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::plonky2::parser::global::{GlobalRecursiveStarkParser, GlobalStarkParser};
use crate::plonky2::parser::{RecursiveStarkParser, StarkParser};
use crate::plonky2::stark::proof::AirProof;
use crate::plonky2::{Plonky2Air, StarkyAir};

#[derive(Debug, Clone)]
pub struct StarkyVerifier<F, C, const D: usize>(core::marker::PhantomData<(F, C)>);

impl<F, C, const D: usize> StarkyVerifier<F, C, D>
where
    F: RichField + Extendable<D>,
    C: CurtaConfig<D, F = F, FE = F::Extension>,
{
    pub fn verify_with_challenges<A>(
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        proof: AirProof<F, C, D>,
        public_inputs: &[F],
        global_values: &[F],
        challenges: StarkProofChallenges<F, D>,
    ) -> Result<()>
    where
        A: StarkyAir<F, D>,
    {
        let degree_bits = config.degree_bits;

        Self::validate_proof_shape(config, stark, &proof, global_values)?;

        let StarkOpeningSet {
            local_values,
            next_values,
            quotient_polys,
        } = &proof.openings;

        // Verify the global constraints
        let mut global_parser = GlobalStarkParser {
            global_vars: global_values,
            public_vars: public_inputs,
            challenges: &challenges.stark_betas,
        };
        stark.air().eval_global(&mut global_parser);

        let global_values_ext = global_values
            .iter()
            .map(|x| F::Extension::from_basefield(*x))
            .collect::<Vec<_>>();
        let public_inputs_ext = public_inputs
            .iter()
            .map(|x| F::Extension::from_basefield(*x))
            .collect::<Vec<_>>();
        let challenges_ext = challenges
            .stark_betas
            .into_iter()
            .map(F::Extension::from_basefield)
            .collect::<Vec<_>>();

        let (l_0, l_last) = Self::eval_l_0_and_l_last(degree_bits, challenges.stark_zeta);
        let last = F::primitive_root_of_unity(degree_bits).inverse();
        let z_last = challenges.stark_zeta - last.into();
        let mut consumer = ConstraintConsumer::<F::Extension>::new(
            challenges
                .stark_alphas
                .iter()
                .map(|&alpha| F::Extension::from_basefield(alpha))
                .collect::<Vec<_>>(),
            z_last,
            l_0,
            l_last,
        );

        let mut parser = StarkParser {
            local_vars: local_values,
            next_vars: next_values,
            global_vars: &global_values_ext,
            public_vars: &public_inputs_ext,
            challenges: &challenges_ext,
            consumer: &mut consumer,
        };

        stark.air().eval(&mut parser);
        let vanishing_polys_zeta = consumer.accumulators();

        // Check each polynomial identity, of the form `vanishing(x) = Z_H(x) quotient(x)`, at zeta.
        let zeta_pow_deg = challenges.stark_zeta.exp_power_of_2(degree_bits);
        let z_h_zeta = zeta_pow_deg - F::Extension::ONE;

        // `quotient_polys_zeta` holds `num_challenges * quotient_degree_factor` evaluations.
        // Each chunk of `quotient_degree_factor` holds the evaluations of `t_0(zeta),...,t_{quotient_degree_factor-1}(zeta)`
        // where the "real" quotient polynomial is `t(X) = t_0(X) + t_1(X)*X^n + t_2(X)*X^{2n} + ...`.
        // So to reconstruct `t(zeta)` we can compute `reduce_with_powers(chunk, zeta^n)` for each
        // `quotient_degree_factor`-sized chunk of the original evaluations.
        let constraint_degree = stark.air().constraint_degree();
        let quotient_degree_factor = 1.max(constraint_degree - 1);
        for (i, chunk) in quotient_polys.chunks(quotient_degree_factor).enumerate() {
            ensure!(
                vanishing_polys_zeta[i] == z_h_zeta * reduce_with_powers(chunk, zeta_pow_deg),
                "Mismatch between evaluation and opening of quotient polynomial"
            );
        }

        let merkle_caps = proof
            .trace_caps
            .into_iter()
            .chain(once(proof.quotient_polys_cap))
            .collect::<Vec<_>>();

        verify_fri_proof::<F, C::GenericConfig, D>(
            &stark.fri_instance(
                challenges.stark_zeta,
                F::primitive_root_of_unity(degree_bits),
                config,
            ),
            &proof.openings.to_fri_openings(),
            &challenges.fri_challenges,
            &merkle_caps,
            &proof.opening_proof,
            &config.fri_params(),
        )?;
        Ok(())
    }

    pub fn verify<A>(
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        proof: StarkProof<F, C, D>,
        public_inputs: &[F],
    ) -> Result<()>
    where
        A: StarkyAir<F, D>,
    {
        let degree_bits = proof.recover_degree_bits(config);
        let challenges = proof.get_challenges(config, stark, public_inputs, degree_bits);
        let StarkProof {
            air_proof,
            global_values,
        } = proof;
        Self::verify_with_challenges(
            config,
            stark,
            air_proof,
            public_inputs,
            &global_values,
            challenges,
        )
    }

    pub fn validate_proof_shape<A: RAirData>(
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        proof: &AirProof<F, C, D>,
        global_values: &[F],
    ) -> Result<()> {
        let fri_params = config.fri_params();
        let cap_height = fri_params.config.cap_height;

        let AirProof {
            trace_caps,
            quotient_polys_cap,
            openings,
            // The shape of the opening proof will be checked in the FRI verifier (see
            // validate_fri_proof_shape), so we ignore it here.
            opening_proof: _,
        } = proof;

        let StarkOpeningSet {
            local_values,
            next_values,
            quotient_polys,
        } = openings;

        for cap in trace_caps.iter() {
            ensure!(cap.height() == cap_height);
        }
        ensure!(quotient_polys_cap.height() == cap_height);
        ensure!(global_values.len() == stark.air().num_global_values());
        ensure!(local_values.len() == stark.air().num_columns());
        ensure!(next_values.len() == stark.air().num_columns());
        ensure!(quotient_polys.len() == stark.num_quotient_polys(config));

        Ok(())
    }

    /// Evaluate the Lagrange polynomials `L_0` and `L_(n-1)` at a point `x`.
    /// `L_0(x) = (x^n - 1)/(n * (x - 1))`
    /// `L_(n-1)(x) = (x^n - 1)/(n * (g * x - 1))`, with `g` the first element of the subgroup.
    fn eval_l_0_and_l_last<FE: Plonky2Field>(log_n: usize, x: FE) -> (FE, FE) {
        let n = FE::from_canonical_usize(1 << log_n);
        let g = FE::primitive_root_of_unity(log_n);
        let z_x = x.exp_power_of_2(log_n) - FE::ONE;
        let invs = FE::batch_multiplicative_inverse(&[n * (x - FE::ONE), n * (g * x - FE::ONE)]);

        (z_x * invs[0], z_x * invs[1])
    }

    pub fn verify_with_challenges_circuit<A>(
        builder: &mut CircuitBuilder<F, D>,
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        proof: &AirProofTarget<D>,
        public_inputs: &[Target],
        global_values: &[Target],
        challenges: StarkProofChallengesTarget<D>,
    ) where
        A: Plonky2Air<F, D>,
    {
        let StarkOpeningSetTarget {
            local_values,
            next_values,
            quotient_polys,
        } = &proof.openings;

        let degree_bits = config.degree_bits;

        let one = builder.one_extension();

        let zeta_pow_deg = builder.exp_power_of_2_extension(challenges.stark_zeta, degree_bits);
        let z_h_zeta = builder.sub_extension(zeta_pow_deg, one);
        let (l_0, l_last) = Self::eval_l_0_and_l_last_circuit(
            builder,
            degree_bits,
            challenges.stark_zeta,
            z_h_zeta,
        );
        let last = builder
            .constant_extension(F::Extension::primitive_root_of_unity(degree_bits).inverse());
        let z_last = builder.sub_extension(challenges.stark_zeta, last);

        let mut consumer = RecursiveConstraintConsumer::<F, D>::new(
            builder.zero_extension(),
            challenges.stark_alphas,
            z_last,
            l_0,
            l_last,
        );

        // verify global constraints
        let mut cubic_results = HashMap::new();
        let mut global_parser = GlobalRecursiveStarkParser {
            builder,
            global_vars: global_values,
            public_vars: public_inputs,
            challenges: &challenges.stark_betas,
            cubic_results: &mut cubic_results,
        };
        stark.air().eval_global(&mut global_parser);

        let global_vals_ext = global_values
            .iter()
            .map(|x| builder.convert_to_ext(*x))
            .collect::<Vec<_>>();
        let public_inputs_ext = public_inputs
            .iter()
            .map(|x| builder.convert_to_ext(*x))
            .collect::<Vec<_>>();
        let challenges_ext = challenges
            .stark_betas
            .iter()
            .map(|x| builder.convert_to_ext(*x))
            .collect::<Vec<_>>();

        let mut parser = RecursiveStarkParser {
            builder,
            local_vars: local_values,
            next_vars: next_values,
            global_vars: &global_vals_ext,
            public_vars: &public_inputs_ext,
            challenges: &challenges_ext,
            consumer: &mut consumer,
        };

        stark.air().eval(&mut parser);

        let vanishing_polys_zeta = consumer.accumulators();

        // Check each polynomial identity, of the form `vanishing(x) = Z_H(x) quotient(x)`, at zeta.
        let mut scale = ReducingFactorTarget::new(zeta_pow_deg);
        let quotient_degree_factor = stark.air().quotient_degree_factor();
        for (i, chunk) in quotient_polys.chunks(quotient_degree_factor).enumerate() {
            let recombined_quotient = scale.reduce(chunk, builder);
            let computed_vanishing_poly = builder.mul_extension(z_h_zeta, recombined_quotient);
            builder.connect_extension(vanishing_polys_zeta[i], computed_vanishing_poly);
        }

        let merkle_caps = proof
            .trace_caps
            .iter()
            .cloned()
            .chain(once(proof.quotient_polys_cap.clone()))
            .collect::<Vec<_>>();

        let fri_instance = stark.fri_instance_target(
            builder,
            challenges.stark_zeta,
            F::primitive_root_of_unity(degree_bits),
            config,
        );
        builder.verify_fri_proof::<C::GenericConfig>(
            &fri_instance,
            &proof.openings.to_fri_openings(),
            &challenges.fri_challenges,
            &merkle_caps,
            &proof.opening_proof,
            &config.fri_params(),
        );
    }

    pub fn verify_circuit<A>(
        builder: &mut CircuitBuilder<F, D>,
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        proof: &StarkProofTarget<D>,
        public_inputs: &[Target],
    ) where
        A: Plonky2Air<F, D>,
    {
        let challenges = proof.get_challenges_target(builder, config, public_inputs, stark);
        let StarkProofTarget {
            air_proof,
            global_values,
        } = proof;
        Self::verify_with_challenges_circuit(
            builder,
            config,
            stark,
            air_proof,
            public_inputs,
            global_values,
            challenges,
        )
    }

    fn eval_l_0_and_l_last_circuit(
        builder: &mut CircuitBuilder<F, D>,
        log_n: usize,
        x: ExtensionTarget<D>,
        z_x: ExtensionTarget<D>,
    ) -> (ExtensionTarget<D>, ExtensionTarget<D>) {
        let n = builder.constant_extension(F::Extension::from_canonical_usize(1 << log_n));
        let g = builder.constant_extension(F::Extension::primitive_root_of_unity(log_n));
        let one = builder.one_extension();
        let l_0_deno = builder.mul_sub_extension(n, x, n);
        let l_last_deno = builder.mul_sub_extension(g, x, one);
        let l_last_deno = builder.mul_extension(n, l_last_deno);

        (
            builder.div_extension(z_x, l_0_deno),
            builder.div_extension(z_x, l_last_deno),
        )
    }
}

pub fn add_virtual_air_proof<
    F: RichField + Extendable<D>,
    A: Plonky2Air<F, D>,
    C: CurtaConfig<D, F = F>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &Starky<A>,
    config: &StarkyConfig<C, D>,
) -> AirProofTarget<D> {
    let fri_params = config.fri_params();
    let cap_height = fri_params.config.cap_height;

    let num_leaves_per_oracle = stark
        .air()
        .round_data()
        .into_iter()
        .map(|x| x.num_columns)
        .chain(once(
            stark.air().quotient_degree_factor() * config.num_challenges,
        ))
        .collect::<Vec<_>>();

    let num_rounds = stark.air().num_rounds();
    let trace_caps = (0..num_rounds)
        .map(|_| builder.add_virtual_cap(cap_height))
        .collect::<Vec<_>>();

    AirProofTarget {
        trace_caps,
        quotient_polys_cap: builder.add_virtual_cap(cap_height),
        openings: add_stark_opening_set_target(builder, stark, config),
        opening_proof: builder.add_virtual_fri_proof(&num_leaves_per_oracle, &fri_params),
    }
}

pub fn add_virtual_stark_proof<
    F: RichField + Extendable<D>,
    A: Plonky2Air<F, D>,
    C: CurtaConfig<D, F = F>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &Starky<A>,
    config: &StarkyConfig<C, D>,
) -> StarkProofTarget<D> {
    let num_global_values = stark.air().num_global_values();
    let global_values_target = builder.add_virtual_targets(num_global_values);
    let air_proof = add_virtual_air_proof(builder, stark, config);
    StarkProofTarget {
        air_proof,
        global_values: global_values_target,
    }
}

pub(crate) fn add_stark_opening_set_target<
    F: RichField + Extendable<D>,
    A: for<'a> RAir<RecursiveStarkParser<'a, F, D>>,
    C: CurtaConfig<D, F = F>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &Starky<A>,
    config: &StarkyConfig<C, D>,
) -> StarkOpeningSetTarget<D> {
    let num_challenges = config.num_challenges;
    StarkOpeningSetTarget {
        local_values: builder.add_virtual_extension_targets(stark.air().num_columns()),
        next_values: builder.add_virtual_extension_targets(stark.air().num_columns()),
        quotient_polys: builder
            .add_virtual_extension_targets(stark.air().quotient_degree_factor() * num_challenges),
    }
}

pub fn set_air_proof_target<F, C: CurtaConfig<D, F = F>, W, const D: usize>(
    witness: &mut W,
    proof_target: &AirProofTarget<D>,
    proof: &AirProof<F, C, D>,
) where
    F: RichField + Extendable<D>,
    C::Hasher: AlgebraicHasher<F>,
    W: WitnessWrite<F>,
{
    for (cap, target_cap) in proof
        .trace_caps
        .iter()
        .zip_eq(proof_target.trace_caps.iter())
    {
        witness.set_cap_target(target_cap, cap);
    }
    witness.set_cap_target(&proof_target.quotient_polys_cap, &proof.quotient_polys_cap);

    witness.set_fri_openings(
        &proof_target.openings.to_fri_openings(),
        &proof.openings.to_fri_openings(),
    );

    set_fri_proof_target(witness, &proof_target.opening_proof, &proof.opening_proof);
}

pub fn set_stark_proof_target<F, C: CurtaConfig<D, F = F>, W, const D: usize>(
    witness: &mut W,
    proof_target: &StarkProofTarget<D>,
    proof: &StarkProof<F, C, D>,
) where
    F: RichField + Extendable<D>,
    C::Hasher: AlgebraicHasher<F>,
    W: WitnessWrite<F>,
{
    set_air_proof_target(witness, &proof_target.air_proof, &proof.air_proof);

    for (target, value) in proof_target
        .global_values
        .iter()
        .zip_eq(proof.global_values.iter())
    {
        witness.set_target(*target, *value);
    }
}
