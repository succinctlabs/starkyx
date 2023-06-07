use plonky2::field::types::Field;

use crate::curta::air::parser::AirParser;
use crate::curta::parameters::FieldParameters;
use crate::curta::polynomial::parser::PolynomialParser;
use crate::curta::polynomial::Polynomial;

pub fn eval_field_operation<AP: AirParser, P: FieldParameters>(
    poly_parser: &mut PolynomialParser<AP>,
    p_vanishing: &Polynomial<AP::Var>,
    p_witness_low: &Polynomial<AP::Var>,
    p_witness_high: &Polynomial<AP::Var>,
) -> Vec<AP::Var> {
    // Reconstruct and shift back the witness polynomial
    let limb = AP::Field::from_canonical_u32(2u32.pow(16));

    let p_witness_high_mul_limb = poly_parser.scalar_mul(p_witness_high, &limb);
    let p_witness_shifted = poly_parser.add(p_witness_low, &p_witness_high_mul_limb);

    // Shift down the witness polynomial. Shifting is needed to range check that each
    // coefficient w_i of the witness polynomial satisfies |w_i| < 2^20.
    let offset = AP::Field::from_canonical_u32(P::WITNESS_OFFSET as u32);
    let p_witness = poly_parser.scalar_sub(&p_witness_shifted, &offset);

    // Multiply by (x-2^16) and make the constraint
    let root_monomiial = Polynomial::from_coefficients(vec![-limb, AP::Field::ONE]);
    let p_witness_mul_root = poly_parser.scalar_poly_mul(&p_witness, &root_monomiial);

    poly_parser
        .sub(p_vanishing, &p_witness_mul_root)
        .coefficients
}
