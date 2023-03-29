use core::marker::PhantomData;

use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::util::transpose;

use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub const N_LIMBS: usize = 16;
pub const NUM_ARITH_COLUMNS: usize = 14 * N_LIMBS;
const RANGE_MAX: usize = 1usize << 16; // Range check strict upper bound

#[derive(Copy, Clone)]
pub struct MulModStark<F, const D: usize> {
    _marker: PhantomData<F>,
}

// Polynomial multiplication
fn poly_mul<F: PrimeField64>(a: &[F], b: &[F]) -> Vec<F> {
    assert_eq!(a.len(), b.len());

    let mut result = vec![F::ZERO; a.len() + b.len()];

    for (i, a) in a.iter().enumerate() {
        for (j, b) in b.iter().enumerate() {
            result[i + j] += *a * *b;
        }
    }
    result
}

type MultiplyTuple = (BigUint, BigUint, BigUint);

impl<F: PrimeField64, const D: usize> MulModStark<F, D> {
    /// Generate trace for addition stark
    fn generate_trace(&self, multiplications: Vec<MultiplyTuple>) -> Vec<PolynomialValues<F>> {
        let max_rows = core::cmp::max(2 * multiplications.len(), RANGE_MAX);
        let mut trace_rows: Vec<Vec<F>> = Vec::with_capacity(max_rows);

        for (a, b, m) in multiplications {
            let row = ArithmeticParser::<F>::mul_to_rows(a, b, m);
            trace_rows.push(row);
        }

        let trace_cols = transpose(&trace_rows);

        trace_cols.into_iter().map(PolynomialValues::new).collect()
    }
}

pub struct ArithmeticParser<F> {
    _marker: PhantomData<F>,
}

impl<F: PrimeField64> ArithmeticParser<F> {
    /// Converts two BigUint inputs into the correspinding rows of addition mod modulus
    ///
    /// a * b = c mod m
    ///
    /// Each element represented by a polynomial a(x), b(x), c(x), m(x) of 16 limbs of 16 bits each
    /// We will witness the relation
    ///  a(x) * b(x) - c(x) - carry(x) * m(x) - (x - 2^16) * s(x) == 0
    /// only a(x), b(x), c(x), m(x) should be range-checked.
    /// where carry = 0 or carry = 1
    /// the first row will contain a(x), b(x), m(x) and the second row will contain c(x), q(x), s(x)
    pub fn mul_to_rows(a_bigint: BigUint, b_bigint: BigUint, m_bigint: BigUint) -> Vec<F> {
        let mut row = vec![F::ZERO; NUM_ARITH_COLUMNS];

        let c_bigint = (&a_bigint * &b_bigint) % &m_bigint;
        debug_assert!(c_bigint < m_bigint);
        let carry_bigint = (&a_bigint * &b_bigint - &c_bigint) / &m_bigint;
        debug_assert!(carry_bigint < m_bigint);

        let a_digits_16_limbs = Self::bigint_into_u16_F_digits(&a_bigint, N_LIMBS);
        let b_digits_16_limbs = Self::bigint_into_u16_F_digits(&b_bigint, N_LIMBS);
        let c_digits_32_limbs = Self::bigint_into_u16_F_digits(&c_bigint, N_LIMBS * 2);
        let carry_digits_16_limbs = Self::bigint_into_u16_F_digits(&carry_bigint, N_LIMBS);
        let m_digits_16_limbs = Self::bigint_into_u16_F_digits(&m_bigint, N_LIMBS);

        let a_mul_b_digits_32_limbs = poly_mul(&a_digits_16_limbs, &b_digits_16_limbs);
        let carry_mul_m_digits_32_limbs = poly_mul(&carry_digits_16_limbs, &m_digits_16_limbs);

        // constr_poly is the array of coefficients of the polynomial
        // a(x) * b(x) - c(x) - carry*m(x) = const(x)
        // note that we don't care about the coefficients of constr(x) at all, just that it will have a root.
        let consr_polynomial_32_limbs: Vec<F> = a_mul_b_digits_32_limbs
            .iter()
            .zip(c_digits_32_limbs.iter())
            .zip(carry_mul_m_digits_32_limbs.iter())
            .map(|((ab, c), cm)| *ab - *c - *cm)
            .collect();

        // assert_eq!(consr_polynomial.len(), N_LIMBS);
        // By assumption β := 2^16 is a root of `a`, i.e. (x - β) divides
        // `a`; if we write
        //
        //    a(x) = \sum_{i=0}^{N-1} a[i] x^i
        //         = (x - β) \sum_{i=0}^{N-2} q[i] x^i
        //
        // then by comparing coefficients it is easy to see that
        //
        //   q[0] = -a[0] / β  and  q[i] = (q[i-1] - a[i]) / β
        //
        //  NOTE : Doing divisions in F::Goldilocks probably not the most efficient
        let mut quotient_digits_32_limbs = Vec::new();
        let two_to_the_16 = F::from_canonical_u32(65536u32);
        quotient_digits_32_limbs.push(-consr_polynomial_32_limbs[0] / two_to_the_16);

        for deg in 1..((N_LIMBS * 2) - 1) {
            let temp1 = quotient_digits_32_limbs[deg - 1];
            let digit = temp1 - consr_polynomial_32_limbs[deg];
            let quot = digit / F::from_canonical_u32(65536u32);
            quotient_digits_32_limbs.push(quot);
        }
        quotient_digits_32_limbs.push(F::ZERO);

        // Add inputs and modulus as values in first row
        for i in 0..N_LIMBS {
            row[i] = a_digits_16_limbs[i];
            row[i + N_LIMBS] = b_digits_16_limbs[i];
            row[i + 2 * N_LIMBS] = carry_digits_16_limbs[i];
            row[i + 3 * N_LIMBS] = m_digits_16_limbs[i];
        }

        // Add result, quotient and aux polynomial as values in second row
        for i in 0..(N_LIMBS * 2) {
            row[i + 4 * N_LIMBS] = a_mul_b_digits_32_limbs[i];
            row[i + 6 * N_LIMBS] = c_digits_32_limbs[i];
            row[i + 8 * N_LIMBS] = carry_mul_m_digits_32_limbs[i];
            row[i + 10 * N_LIMBS] = quotient_digits_32_limbs[i];
        }

        row
    }

    pub fn bigint_into_u16_digits(x: &BigUint) -> Vec<u16> {
        x.iter_u32_digits()
            .flat_map(|x| vec![x as u16, (x >> 16) as u16])
            .collect()
    }

    #[allow(non_snake_case)]
    pub fn bigint_into_u16_F_digits(x: &BigUint, digits: usize) -> Vec<F> {
        let mut x_limbs: Vec<_> = Self::bigint_into_u16_digits(x)
            .iter()
            .map(|xi| F::from_canonical_u16(*xi))
            .collect();
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
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MulModStark<F, D> {
    const COLUMNS: usize = NUM_ARITH_COLUMNS;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: plonky2::field::extension::FieldExtension<D2, BaseField = F>,
        P: plonky2::field::packed::PackedField<Scalar = FE>,
    {
        // // we want to constrain a(x) * b(x) - c(x) - carry(x) * m(x) - (x - β) * s(x) == 0
        // // constrain a(x) * b(x) = a_mul_b(x) and carry(x) * m(x) = carry_mul_m(x);
        let a_offset = 0;
        let b_offset = N_LIMBS;
        let carry_offset = 2 * N_LIMBS;
        let m_offset = 3 * N_LIMBS;
        let a_mul_b_offset = 4 * N_LIMBS;
        let carry_mul_m_offset = 8 * N_LIMBS;

        let mut a_mul_b_degree_terms = [P::ZEROS; N_LIMBS * 2];
        let mut carry_mul_m_terms = [P::ZEROS; N_LIMBS * 2];
        for i in 0..N_LIMBS {
            for j in 0..N_LIMBS {
                let deg = i + j;
                a_mul_b_degree_terms[deg] +=
                    vars.local_values[a_offset + i] * vars.local_values[b_offset + j];
                carry_mul_m_terms[deg] +=
                    vars.local_values[carry_offset + i] * vars.local_values[m_offset + j];
            }
        }
        for i in 0..(N_LIMBS * 2) {
            yield_constr.constraint_transition(
                a_mul_b_degree_terms[i] - vars.local_values[a_mul_b_offset + i],
            );
            yield_constr.constraint_transition(
                carry_mul_m_terms[i] - vars.local_values[carry_mul_m_offset + i],
            );
        }

        // // (x - β) * s(x) == 0
        let mut consr_poly = vec![];
        let pow_2 = P::Scalar::from_canonical_u32(65536u32);

        // // -β * s(x), for x = \omega
        consr_poly.push(-vars.local_values[10 * N_LIMBS].mul(pow_2));

        // // x * s(x) - β * s(x) for x = \omega_i
        for i in 1..(2 * N_LIMBS) - 1 {
            let val = -vars.local_values[i + 10 * N_LIMBS].mul(pow_2)
                + vars.local_values[i - 1 + 10 * N_LIMBS];
            consr_poly.push(val)
        }
        // x * s(x) for x = \omega_n, but this is just 0, because s(x) is degree n - 1
        consr_poly.push(vars.local_values[2 * N_LIMBS - 2 + 10 * N_LIMBS]);

        // // a_mul_b(x) - c(x) - carry_mul_m(x)
        let c_offset = 6 * N_LIMBS;
        for (i, consr) in consr_poly.iter().enumerate().take(N_LIMBS * 2) {
            yield_constr.constraint_transition(
                vars.local_values[a_mul_b_offset + i]
                    - vars.local_values[c_offset + i]
                    - vars.local_values[carry_mul_m_offset + i]
                    - *consr,
            );
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let a_offset = 0;
        let b_offset = N_LIMBS;
        let carry_offset = 2 * N_LIMBS;
        let m_offset = 3 * N_LIMBS;
        let a_mul_b_offset = 4 * N_LIMBS;
        let carry_mul_m_offset = 8 * N_LIMBS;

        let mut a_mul_b_degree_terms = [builder.zero_extension(); N_LIMBS * 2];
        let mut carry_mul_m_terms = [builder.zero_extension(); N_LIMBS * 2];
        for i in 0..N_LIMBS * 2 {
            a_mul_b_degree_terms[i] = builder.zero_extension();
            carry_mul_m_terms[i] = builder.zero_extension();
        }

        for i in 0..N_LIMBS {
            for j in 0..N_LIMBS {
                let deg = i + j;
                let t1 = builder.mul_extension(
                    vars.local_values[a_offset + i],
                    vars.local_values[b_offset + j],
                );
                a_mul_b_degree_terms[deg] = builder.add_extension(t1, a_mul_b_degree_terms[deg]);
                let t2 = builder.mul_extension(
                    vars.local_values[carry_offset + i],
                    vars.local_values[m_offset + j],
                );
                carry_mul_m_terms[deg] = builder.add_extension(t2, carry_mul_m_terms[deg]);
            }
        }
        for i in 0..(N_LIMBS * 2) {
            let x = builder.sub_extension(
                a_mul_b_degree_terms[i],
                vars.local_values[a_mul_b_offset + i],
            );
            yield_constr.constraint_transition(builder, x);
            let y = builder.sub_extension(
                carry_mul_m_terms[i],
                vars.local_values[carry_mul_m_offset + i],
            );
            yield_constr.constraint_transition(builder, y);
        }

        // (x - β) * s(x) == 0
        let mut consr_poly = vec![];
        let pow_2 = builder.constant_extension(F::Extension::from_canonical_u32(65536u32));

        // // // -β * s(x), for x = \omega
        let t = builder.mul_extension(vars.local_values[10 * N_LIMBS], pow_2);
        let neg_one = builder.neg_one_extension();
        consr_poly.push(builder.mul_extension(neg_one, t));

        // // x * s(x) - β * s(x) for x = \omega_i
        for i in 1..(2 * N_LIMBS) - 1 {
            let x = builder.mul_extension(neg_one, vars.local_values[i + 10 * N_LIMBS]);
            let y = builder.mul_extension(x, pow_2);
            let z = builder.add_extension(y, vars.local_values[i - 1 + 10 * N_LIMBS]);
            consr_poly.push(z)
        }
        // // x * s(x) for x = \omega_n, but this is just 0, because s(x) is degree n - 1
        consr_poly.push(vars.local_values[2 * N_LIMBS - 2 + 10 * N_LIMBS]);

        // a_mul_b(x) - c(x) - carry_mul_m(x)
        let c_offset = 6 * N_LIMBS;

        for (i, consr) in consr_poly.iter().enumerate().take(N_LIMBS * 2) {
            let x = builder.sub_extension(
                vars.local_values[a_mul_b_offset + i],
                vars.local_values[c_offset + i],
            );
            let y = builder.sub_extension(x, vars.local_values[carry_mul_m_offset + i]);
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

        let num_rows = 64;

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

        // Verify recursive proof in a circuit
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
