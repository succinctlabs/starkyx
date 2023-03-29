use core::marker::PhantomData;

use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::util::transpose;

use crate::arithmetic::polynomial::PolynomialOps;
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub const NB_LIMBS: usize = 16;
pub const NB_EXPANDED_LIMBS: usize = NB_LIMBS * 2 - 1;
const A_LIMBS_START: usize = 0;
const B_LIMBS_START: usize = A_LIMBS_START + NB_LIMBS;
const C_LIMBS_START: usize = B_LIMBS_START + NB_LIMBS;
const CARRY_LIMBS_START: usize = C_LIMBS_START + NB_LIMBS;
const MODULUS_LIMBS_START: usize = CARRY_LIMBS_START + NB_LIMBS;
const QUOTIENT_LIMBS_START: usize = MODULUS_LIMBS_START + NB_LIMBS;
const NB_COLUMNS: usize = QUOTIENT_LIMBS_START + NB_EXPANDED_LIMBS;
const RANGE_MAX: usize = 1usize << 16;

pub fn bigint_into_u16_digits(x: &BigUint) -> Vec<u16> {
    x.iter_u32_digits()
        .flat_map(|x| vec![x as u16, (x >> 16) as u16])
        .collect()
}

pub fn bigint_to_u16_limbs<F: RichField>(x: &BigUint, digits: usize) -> Vec<F> {
    let mut x_limbs = bigint_into_u16_digits(x)
        .iter()
        .map(|xi| F::from_canonical_u16(*xi))
        .collect::<Vec<F>>();
    assert!(
        x_limbs.len() <= digits,
        "Number too large to fit in {} digits",
        digits
    );
    for _ in x_limbs.len()..digits {
        x_limbs.push(F::ZERO);
    }
    x_limbs
}

#[derive(Copy, Clone)]
pub struct MulModStark<F, const D: usize> {
    _marker: PhantomData<F>,
}

type MultiplyTuple = (BigUint, BigUint, BigUint);

impl<F: RichField, const D: usize> MulModStark<F, D> {
    fn generate_trace(&self, multiplications: Vec<MultiplyTuple>) -> Vec<PolynomialValues<F>> {
        let max_rows = core::cmp::max(2 * multiplications.len(), RANGE_MAX);
        let mut trace_rows: Vec<Vec<F>> = Vec::with_capacity(max_rows);

        for (a, b, m) in multiplications {
            let row = ArithmeticParser::<F>::mul_to_rows(&a, &b, &m);
            trace_rows.push(row);
        }

        let trace_cols = transpose(&trace_rows);

        trace_cols.into_iter().map(PolynomialValues::new).collect()
    }
}

pub struct ArithmeticParser<F> {
    _marker: PhantomData<F>,
}

impl<F: RichField> ArithmeticParser<F> {
    pub fn mul_to_rows(a: &BigUint, b: &BigUint, modulus: &BigUint) -> Vec<F> {
        // Calculate witnesses.
        let c = (a * b) % modulus;
        let carry = (a * b - &c) / modulus;

        // Calculate polynomial representation of big integers.
        let a_limbs = bigint_to_u16_limbs::<F>(a, NB_LIMBS);
        let b_limbs = bigint_to_u16_limbs::<F>(b, NB_LIMBS);
        let c_limbs = bigint_to_u16_limbs::<F>(&c, NB_LIMBS);
        let carry_limbs = bigint_to_u16_limbs::<F>(&carry, NB_LIMBS);
        let modulus_limbs = bigint_to_u16_limbs::<F>(modulus, NB_LIMBS);

        // Calculate polynomial representation of expanded big integers.
        let a_mul_b_limbs_expanded = PolynomialOps::mul(&a_limbs, &b_limbs);
        let carry_mul_modulus_limbs_expanded = PolynomialOps::mul(&carry_limbs, &modulus_limbs);

        // Calculate the constraint polynomial.
        //     constraint(x) = a(x) * b(x) - c(x) - carry(x) * modulus(x)
        // Note that we do not care about the coefficients of the constraint polynomial, we just
        // care that it has a root at (x - β), where β = 2^16.
        let mut constraint_poly_limbs_expanded = Vec::new();
        for i in 0..(2 * NB_LIMBS - 1) {
            let mut value = a_mul_b_limbs_expanded[i] - carry_mul_modulus_limbs_expanded[i];
            if i < NB_LIMBS {
                value -= c_limbs[i];
            }
            constraint_poly_limbs_expanded.push(value)
        }

        // Calculate the quotient polynomaial that proves that constraint(x) has a root at 2^16.
        // β := 2^16 is a root of `a` if (x - β) divides `a`; if we write
        //
        //    a(x) = \sum_{i=0}^{N-1} a[i] x^i
        //         = (x - β) \sum_{i=0}^{N-2} q[i] x^i
        //
        // then by comparing coefficients it is easy to see that
        //
        //   q[0] = -a[0] / β  and  q[i] = (q[i-1] - a[i]) / β
        //
        let beta = F::from_canonical_u32(1 << 16);
        let mut quotient_limbs_expanded = Vec::new();
        quotient_limbs_expanded.push(-constraint_poly_limbs_expanded[0] / beta);
        for i in 1..((NB_LIMBS * 2) - 1) {
            let coefficient = quotient_limbs_expanded[i - 1] - constraint_poly_limbs_expanded[i];
            let quotient = coefficient / beta;
            quotient_limbs_expanded.push(quotient);
        }
        quotient_limbs_expanded.push(F::ZERO);

        // Write computed values to trace table.
        let mut row = vec![F::ZERO; NB_COLUMNS];
        for i in 0..NB_LIMBS {
            row[A_LIMBS_START + i] = a_limbs[i];
            row[B_LIMBS_START + i] = b_limbs[i];
            row[C_LIMBS_START + i] = c_limbs[i];
            row[CARRY_LIMBS_START + i] = carry_limbs[i];
            row[MODULUS_LIMBS_START + i] = modulus_limbs[i];
        }
        for i in 0..NB_EXPANDED_LIMBS {
            row[QUOTIENT_LIMBS_START + i] = quotient_limbs_expanded[i];
        }
        row
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MulModStark<F, D> {
    const COLUMNS: usize = NB_COLUMNS;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: plonky2::field::extension::FieldExtension<D2, BaseField = F>,
        P: plonky2::field::packed::PackedField<Scalar = FE>,
    {
        // We want to constrain a(x) * b(x) - c(x) - carry(x) * m(x) - (x - β) * s(x) == 0.
        // Compute product terms a(x) * b(x) and carry(x) * m(x).
        let mut a_mul_b_limbs_expanded = [P::ZEROS; NB_EXPANDED_LIMBS];
        let mut carry_mul_m_limbs_expanded = [P::ZEROS; NB_EXPANDED_LIMBS];
        for i in 0..NB_LIMBS {
            for j in 0..NB_LIMBS {
                let degree = i + j;
                a_mul_b_limbs_expanded[degree] +=
                    vars.local_values[A_LIMBS_START + i] * vars.local_values[B_LIMBS_START + j];
                carry_mul_m_limbs_expanded[degree] += vars.local_values[CARRY_LIMBS_START + i]
                    * vars.local_values[MODULUS_LIMBS_START + j];
            }
        }

        // Compute the polynomial constraint term (x - β) * s(x). Note that s(x) has degree 2N - 2
        // because s(x) is one less degree than the constraint polynomial. Note that this expression
        // expands out to: x * s(x) - β * s(x).
        let mut consr_poly = vec![];
        let beta = P::Scalar::from_canonical_u32(1 << 16);

        // x * s(x) - β * s(x) = β * s(x) for x = \omega_0
        consr_poly.push(-vars.local_values[QUOTIENT_LIMBS_START].mul(beta));

        // x * s(x) - β * s(x) for x = \omega_i, where i \neq 0 or n - 1
        for i in 1..NB_EXPANDED_LIMBS {
            let val = -vars.local_values[QUOTIENT_LIMBS_START + i].mul(beta)
                + vars.local_values[QUOTIENT_LIMBS_START + i - 1];
            consr_poly.push(val)
        }

        // x * s(x) - β * s(x) = x * s(x) for x = \omega_{n-1}
        consr_poly.push(vars.local_values[QUOTIENT_LIMBS_START + NB_EXPANDED_LIMBS - 1]);

        // Final Constraint: a(x) * b(x) - c(x) - carry(x) * mul(x) - (x - \beta) * s(x) == 0
        for (i, consr) in consr_poly.iter().enumerate().take(NB_EXPANDED_LIMBS) {
            if i < NB_LIMBS {
                yield_constr.constraint_transition(
                    a_mul_b_limbs_expanded[i]
                        - vars.local_values[C_LIMBS_START + i]
                        - carry_mul_m_limbs_expanded[i]
                        - *consr,
                );
            } else {
                yield_constr.constraint_transition(
                    a_mul_b_limbs_expanded[i] - carry_mul_m_limbs_expanded[i] - *consr,
                );
            }
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let zero = builder.zero_extension();
        let mut a_mul_b_limbs_expanded = [zero; NB_EXPANDED_LIMBS];
        let mut carry_mul_m_limbs_expanded = [zero; NB_EXPANDED_LIMBS];

        // We want to constrain a(x) * b(x) - c(x) - carry(x) * m(x) - (x - β) * s(x) == 0.
        // Compute product terms a(x) * b(x) and carry(x) * m(x).
        for i in 0..NB_LIMBS {
            for j in 0..NB_LIMBS {
                let degree = i + j;
                let a_mul_b_term = builder.mul_extension(
                    vars.local_values[A_LIMBS_START + i],
                    vars.local_values[B_LIMBS_START + j],
                );
                a_mul_b_limbs_expanded[degree] =
                    builder.add_extension(a_mul_b_term, a_mul_b_limbs_expanded[degree]);
                let carry_mul_m_term = builder.mul_extension(
                    vars.local_values[CARRY_LIMBS_START + i],
                    vars.local_values[MODULUS_LIMBS_START + j],
                );
                carry_mul_m_limbs_expanded[degree] =
                    builder.add_extension(carry_mul_m_term, carry_mul_m_limbs_expanded[degree]);
            }
        }

        // Compute the polynomial constraint term (x - β) * s(x). Note that s(x) has degree 2N - 2
        // because s(x) is one less degree than the constraint polynomial. Note that this expression
        // expands out to: x * s(x) - β * s(x).
        let mut consr_poly = vec![];
        let beta = builder.constant_extension(F::Extension::from_canonical_u32(1 << 16));

        // x * s(x) - β * s(x) = β * s(x) for x = \omega_0
        let beta_mul_s = builder.mul_extension(vars.local_values[QUOTIENT_LIMBS_START], beta);
        let neg_one = builder.neg_one_extension();
        consr_poly.push(builder.mul_extension(neg_one, beta_mul_s));

        // x * s(x) - β * s(x) for x = \omega_i, where i \neq 0 or n - 1
        for i in 1..NB_EXPANDED_LIMBS {
            let neg_s = builder.mul_extension(neg_one, vars.local_values[QUOTIENT_LIMBS_START + i]);
            let neg_beta_mul_s = builder.mul_extension(neg_s, beta);
            consr_poly.push(builder.add_extension(
                neg_beta_mul_s,
                vars.local_values[QUOTIENT_LIMBS_START + i - 1],
            ));
        }

        // x * s(x) - β * s(x) = x * s(x) for x = \omega_{n-1}
        consr_poly.push(vars.local_values[QUOTIENT_LIMBS_START + NB_EXPANDED_LIMBS - 1]);

        // Final Constraint: a(x) * b(x) - c(x) - carry(x) * mul(x) - (x - \beta) * s(x) == 0
        for (i, consr) in consr_poly.iter().enumerate().take(NB_EXPANDED_LIMBS) {
            let x = if i < NB_LIMBS {
                builder.sub_extension(
                    a_mul_b_limbs_expanded[i],
                    vars.local_values[C_LIMBS_START + i],
                )
            } else {
                a_mul_b_limbs_expanded[i]
            };
            let y = builder.sub_extension(x, carry_mul_m_limbs_expanded[i]);
            let z = builder.sub_extension(y, *consr);
            yield_constr.constraint_transition(builder, z);
        }
    }

    fn constraint_degree(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {

    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[test]
    fn test_mul_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = MulModStark<F, D>;

        let num_rows = 8192;

        let config = StarkConfig::standard_fast_config();

        let stark = S {
            _marker: PhantomData,
        };

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        let mut additions = Vec::new();
        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(255) % &p22519;
            let b = rng.gen_biguint(255) % &p22519;
            let p = p22519.clone();
            additions.push((a, b, p));
        }

        let trace = stark.generate_trace(additions);

        // Verify proof as a stark
        let proof =
            prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default()).unwrap();
        verify_stark_proof(stark, proof.clone(), &config).unwrap();

        // // Verify recursive proof in a circuit
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof =
            add_virtual_stark_proof_with_pis(&mut recursive_builder, stark, &config, degree_bits);

        recursive_builder.print_gate_counts(0);

        let mut rec_pw = PartialWitness::new();
        set_stark_proof_with_pis_target(&mut rec_pw, &virtual_proof, &proof);

        verify_stark_proof_circuit::<F, C, S, D>(
            &mut recursive_builder,
            stark,
            virtual_proof,
            &config,
        );

        let recursive_data = recursive_builder.build::<C>();

        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut timing,
        )
        .unwrap();

        // timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }
}
