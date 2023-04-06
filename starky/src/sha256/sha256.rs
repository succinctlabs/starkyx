//! Implementation of modular multiplication as a STARK (prototype)
//!
//! The implementation is based on a method used in Polygon starks

use core::marker::PhantomData;

use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
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

#[derive(Copy, Clone)]
pub struct Sha256Stark<F, const D: usize> {
    _marker: PhantomData<F>,
}

impl<F: RichField, const D: usize> MulModStark<F, D> {
    fn generate_trace(&self) -> Vec<PolynomialValues<F>> {
        let max_rows = core::cmp::max(2 * multiplications.len(), RANGE_MAX);
        let mut trace_rows: Vec<Vec<F>> = Vec::with_capacity(max_rows);

        let trace_cols = transpose(&trace_rows);

        trace_cols.into_iter().map(PolynomialValues::new).collect()
    }
}
