use num::BigUint;

use super::parameters::FieldParameters;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;
use crate::polynomial::Polynomial;

pub fn eval_field_operation<AP: PolynomialParser, P: FieldParameters>(
    parser: &mut AP,
    p_vanishing: &Polynomial<AP::Var>,
    p_witness_low: &Polynomial<AP::Var>,
    p_witness_high: &Polynomial<AP::Var>,
) {
    // Reconstruct and shift back the witness polynomial
    let limb_field = AP::Field::from_canonical_u32(2u32.pow(16));
    let limb = parser.constant(limb_field);

    let p_witness_high_mul_limb = parser.poly_scalar_mul(p_witness_high, &limb);
    let p_witness_shifted = parser.poly_add(p_witness_low, &p_witness_high_mul_limb);

    // Shift down the witness polynomial. Shifting is needed to range check that each
    // coefficient w_i of the witness polynomial satisfies |w_i| < 2^20.
    let offset = AP::Field::from_canonical_u32(P::WITNESS_OFFSET as u32);
    let offset = parser.constant(offset);
    let p_witness = parser.poly_scalar_sub(&p_witness_shifted, &offset);

    // Multiply by (x-2^16) and make the constraint
    let root_monomial = Polynomial::from_coefficients(vec![-limb_field, AP::Field::ONE]);
    let p_witness_mul_root = parser.poly_mul_poly_const(&p_witness, &root_monomial);

    let constraints = parser.poly_sub(p_vanishing, &p_witness_mul_root);
    for constr in constraints.coefficients {
        parser.constraint(constr);
    }
}

pub fn modulus_field_iter<F: Field, P: FieldParameters>() -> impl Iterator<Item = F> {
    P::MODULUS
        .into_iter()
        .map(|x| F::from_canonical_u16(x))
        .take(P::NB_LIMBS)
}

#[inline]
pub fn compute_root_quotient_and_shift<F: Field>(
    p_vanishing: &Polynomial<F>,
    offset: usize,
) -> Vec<F> {
    // Evaluate the vanishing polynomial at x = 2^16.
    let p_vanishing_eval = p_vanishing
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, x)| F::from_noncanonical_biguint(BigUint::from(2u32).pow(16 * i as u32)) * *x)
        .sum::<F>();
    debug_assert_eq!(p_vanishing_eval, F::ZERO);

    // Compute the witness polynomial by witness(x) = vanishing(x) / (x - 2^16).
    let root_monomial = F::from_canonical_u32(2u32.pow(16));
    let p_quotient = p_vanishing.root_quotient(root_monomial);
    debug_assert_eq!(p_quotient.degree(), p_vanishing.degree() - 1);

    // Sanity Check #1: For all i, |w_i| < 2^20 to prevent overflows.
    let offset_u64 = offset as u64;

    // Sanity Check #2: w(x) * (x - 2^16) = vanishing(x).
    let x_minus_root = Polynomial::<F>::from_coefficients_slice(&[-root_monomial, F::ONE]);
    debug_assert_eq!(
        (&p_quotient * &x_minus_root).coefficients(),
        p_vanishing.coefficients()
    );

    // Shifting the witness polynomial to make it positive
    p_quotient
        .coefficients()
        .iter()
        .map(|x| *x + F::from_canonical_u64(offset_u64))
        .collect::<Vec<F>>()
}
