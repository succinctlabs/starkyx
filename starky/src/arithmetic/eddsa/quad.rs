use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::*;
use crate::arithmetic::{ArithmeticOp, ArithmeticParser, Register};

pub const N_LIMBS: usize = 16;
pub const NUM_CARRY_LIMBS: usize = 2 * N_LIMBS + 1;
pub const NUM_WITNESS_LIMBS: usize = 2 * N_LIMBS;
const WITNESS_OFFSET: usize = 1usize << 20; // Witness offset

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
    /// [output[N_LIMBS], carry[NUM_CARRY_LIMBS], Witness_low[NUM_WITNESS_LIMBS], Witness_high[NUM_WITNESS_LIMBS]]
    pub fn quad_trace(a: BigUint, b: BigUint, c: BigUint, d: BigUint) -> Vec<F> {
        let p = get_p();
        let result = (&a * &b + &c * &d) % &p;
        debug_assert!(result < p);
        let carry = (&a * &b + &c * &d - &result) / &p;
        debug_assert!(carry < 2u32 * &p);
        debug_assert_eq!(&carry * &p, &a * &b + &c * &d - &result);

        vec![]
    }
}
