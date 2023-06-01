use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::curta::air::parser::AirParser;
use crate::curta::parameters::{FieldParameters, LIMB};
use crate::curta::polynomial::parser::PolynomialParser;
use crate::curta::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};

pub fn packed_generic_field_operation<
    F: RichField + Extendable<D>,
    const D: usize,
    FE,
    PF,
    const D2: usize,
    P: FieldParameters,
>(
    p_vanishing: Vec<PF>,
    p_witness_low: &[PF],
    p_witness_high: &[PF],
) -> Vec<PF>
where
    FE: FieldExtension<D2, BaseField = F>,
    PF: PackedField<Scalar = FE>,
{
    // Compute the witness polynomial using the automatically range checked columns.
    let limb = FE::from_canonical_u32(LIMB);
    let p_witness_shifted = p_witness_low
        .iter()
        .zip(p_witness_high.iter())
        .map(|(x, y)| *x + (*y * limb));

    // Shift down the witness polynomial. Shifting is needed to range check that each
    // coefficient w_i of the witness polynomial satisfies |w_i| < 2^20.
    let offset = FE::from_canonical_u32(P::WITNESS_OFFSET as u32);
    let w = p_witness_shifted.map(|x| x - offset).collect::<Vec<PF>>();

    // Multiply by (x-2^16) and make the constraint
    let root_monomial: &[PF] = &[PF::from(-limb), PF::from(PF::Scalar::ONE)];
    let witness_times_root = PolynomialOps::mul(&w, root_monomial);

    p_vanishing
        .iter()
        .zip(witness_times_root.iter())
        .map(|(x, y)| *x - *y)
        .collect()
}

pub fn ext_circuit_field_operation<
    F: RichField + Extendable<D>,
    const D: usize,
    P: FieldParameters,
>(
    builder: &mut CircuitBuilder<F, D>,
    p_vanishing: Vec<ExtensionTarget<D>>,
    p_witness_low: &[ExtensionTarget<D>],
    p_witness_high: &[ExtensionTarget<D>],
) -> Vec<ExtensionTarget<D>> {
    type PG = PolynomialGadget;

    // Reconstruct and shift back the witness polynomial
    let limb_fe = F::Extension::from_canonical_u32(2u32.pow(16));
    let limb = builder.constant_extension(limb_fe);
    let p_witness_high_mul_limb = PG::ext_scalar_mul_extension(builder, p_witness_high, &limb);
    let p_witness_shifted = PG::add_extension(builder, p_witness_low, &p_witness_high_mul_limb);

    // Shift down the witness polynomial. Shifting is needed to range check that each
    // coefficient w_i of the witness polynomial satisfies |w_i| < 2^20.
    let offset_fe = F::Extension::from_canonical_u32(P::WITNESS_OFFSET as u32);
    let offset = builder.constant_extension(offset_fe);
    let p_witness = PG::sub_constant_extension(builder, &p_witness_shifted, &offset);

    // Multiply by (x-2^16) and make the constraint
    let neg_limb = builder.constant_extension(-limb_fe);
    let root_monomial = &[neg_limb, builder.constant_extension(F::Extension::ONE)];
    let p_witness_mul_root = PG::mul_extension(builder, &p_witness.as_slice(), root_monomial);

    // Check [a(x) + b(x) - result(x) - carry(x) * p(x)] - [witness(x) * (x-2^16)] = 0.
    PG::sub_extension(builder, &p_vanishing, &p_witness_mul_root)
}

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
        .sub(&p_vanishing, &p_witness_mul_root)
        .coefficients
}
