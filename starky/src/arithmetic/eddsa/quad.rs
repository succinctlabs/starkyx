use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::*;
use crate::arithmetic::polynomial::Polynomial;
use crate::arithmetic::util::{extract_witness_and_shift, split_digits, to_field_iter};
use crate::arithmetic::{ArithmeticOp, ArithmeticParser, Register};

pub const N_LIMBS: usize = 16;
pub const NUM_CARRY_LIMBS: usize = N_LIMBS;
pub const NUM_WITNESS_LIMBS: usize = 2 * N_LIMBS - 3;
const WITNESS_OFFSET: usize = 1usize << 20; // Witness offset
const NUM_QUAD_COLUMNS: usize = 4 * N_LIMBS + NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS;

/// A gadget to compute
/// QUAD(x, y, z, w) = (a * b + c * d) mod p
#[derive(Debug, Clone, Copy)]
pub struct QuadLayout {
    a: Register,
    b: Register,
    c: Register,
    d: Register,
    output: Register,
    witness: Register,
}

impl QuadLayout {
    #[inline]
    pub const fn new(
        a: Register,
        b: Register,
        c: Register,
        d: Register,
        output: Register,
        witness: Register,
    ) -> Self {
        Self {
            a,
            b,
            c,
            d,
            output,
            witness,
        }
    }

    #[inline]
    pub fn carry_range(&self) -> (usize, usize) {
        let start = self.witness.index();
        (start, start + NUM_CARRY_LIMBS)
    }

    #[inline]
    pub fn witness_low_range(&self) -> (usize, usize) {
        let start = self.witness.index();
        (
            start + NUM_CARRY_LIMBS,
            start + NUM_CARRY_LIMBS + NUM_WITNESS_LIMBS,
        )
    }

    #[inline]
    pub fn witness_high_range(&self) -> (usize, usize) {
        let start = self.witness.index();
        (
            start + NUM_CARRY_LIMBS + NUM_WITNESS_LIMBS,
            start + NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS,
        )
    }
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    /// Returns a vector
    /// [Input[4 * N_LIMBS], output[N_LIMBS], carry[NUM_CARRY_LIMBS], Witness_low[NUM_WITNESS_LIMBS], Witness_high[NUM_WITNESS_LIMBS]]
    pub fn quad_trace(a: BigUint, b: BigUint, c: BigUint, d: BigUint) -> Vec<F> {
        let p = get_p();
        let result = (&a * &b + &c * &d) % &p;
        debug_assert!(result < p);
        let carry = (&a * &b + &c * &d - &result) / &p;
        debug_assert!(carry < 2u32 * &p);
        debug_assert_eq!(&carry * &p, &a * &b + &c * &d - &result);

        // make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(&a, 16, N_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(&b, 16, N_LIMBS);
        let p_c = Polynomial::<i64>::from_biguint_num(&c, 16, N_LIMBS);
        let p_d = Polynomial::<i64>::from_biguint_num(&d, 16, N_LIMBS);
        let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, N_LIMBS);

        let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
        let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, NUM_CARRY_LIMBS);

        // Compute the vanishing polynomial
        let vanishing_poly = &p_a * &p_b + &p_c * &p_d - &p_result - &p_carry * &p_p;
        debug_assert_eq!(vanishing_poly.degree(), NUM_WITNESS_LIMBS + 1);

        // Compute the witness
        let witness_shifted = extract_witness_and_shift(&vanishing_poly, WITNESS_OFFSET as u32);
        let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

        let mut row = Vec::with_capacity(NUM_QUAD_COLUMNS);

        // inputs
        row.extend(to_field_iter::<F>(&p_a));
        row.extend(to_field_iter::<F>(&p_b));
        row.extend(to_field_iter::<F>(&p_c));
        row.extend(to_field_iter::<F>(&p_d));

        // output
        row.extend(to_field_iter::<F>(&p_result));
        // carry and witness
        row.extend(to_field_iter::<F>(&p_carry));
        row.extend(witness_low);
        row.extend(witness_high);

        row
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use num::bigint::RandBigInt;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn test_quad_trace_generation() {
        let num_tests = 100;
        let p = get_p();
        const D : usize = 2;
        type F = <PoseidonGoldilocksConfig as GenericConfig<D>>::F;

        for _ in 0..num_tests {
            let a = rand::thread_rng().gen_biguint(256) % &p;
            let b = rand::thread_rng().gen_biguint(256) & &p;
            let c = rand::thread_rng().gen_biguint(256) & &p;
            let d = rand::thread_rng().gen_biguint(256) & &p;

            let _ = ArithmeticParser::<F, 4>::quad_trace(a.clone(), b.clone(), c.clone(), d.clone());
        }

    }
}