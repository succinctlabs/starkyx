use num::{BigUint, Zero};

use super::point::{AffinePoint, AffinePointRegister};
use super::{EllipticCurve, EllipticCurveParameters};
use crate::chip::builder::AirBuilder;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::parameters::{FieldParameters, MAX_NB_LIMBS};
use crate::chip::AirParameters;

pub mod add;
pub mod bigint_operations;
pub mod ed25519;
pub mod scalar_mul;

pub trait EdwardsParameters: EllipticCurveParameters {
    const D: [u16; MAX_NB_LIMBS];

    fn generator() -> AffinePoint<Self>;

    fn prime_group_order() -> BigUint;

    fn d_biguint() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::D.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }

    fn nb_scalar_bits() -> usize {
        Self::BaseField::NB_LIMBS * 16
    }

    fn neutral() -> AffinePoint<Self> {
        AffinePoint::new(BigUint::from(0u32), BigUint::from(1u32))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EdwardsCurve<E: EdwardsParameters>(pub E);

impl<E: EdwardsParameters> EllipticCurveParameters for EdwardsCurve<E> {
    type BaseField = E::BaseField;
}

impl<L: AirParameters, E: EdwardsParameters> EllipticCurve<L> for EdwardsCurve<E>
where
    L::Instruction: FromFieldInstruction<E::BaseField>,
{
    fn ec_add(
        builder: &mut AirBuilder<L>,
        p: &AffinePointRegister<Self>,
        q: &super::point::AffinePointRegister<Self>,
    ) -> super::point::AffinePointRegister<Self> {
        // Ed25519 Elliptic Curve Addition Formula
        //
        // Given two elliptic curve points (x1, y1) and (x2, y2), compute the sum (x3, y3) with
        //
        // x3 = (x1 * y2 + x2 * y1) / (1 + d * f)
        // y3 = (y1 * y2 + x1 * x2) / (1 - d * f)
        //
        // where f = x1 * x2 * y1 * y2.
        //
        // Reference: https://datatracker.ietf.org/doc/html/draft-josefsson-eddsa-ed25519-02

        let x1 = p.x;
        let x2 = q.x;
        let y1 = p.y;
        let y2 = q.y;

        // x3_numerator = x1 * y2 + x2 * y1.
        let x3_numerator = builder.fp_inner_product(&vec![x1, x2], &vec![y2, y1]);

        // y3_numerator = y1 * y2 + x1 * x2.
        let y3_numerator = builder.fp_inner_product(&vec![y1, x1], &vec![y2, x2]);

        // f = x1 * x2 * y1 * y2.
        let x1_mul_y1 = builder.fp_mul(&x1, &y1);
        let x2_mul_y2 = builder.fp_mul(&x2, &y2);
        let f = builder.fp_mul(&x1_mul_y1, &x2_mul_y2);

        // d * f.
        let d_mul_f = builder.fp_mul_const(&f, E::D);

        // x3 = x3_numerator / (1 + d * f).
        let x3_ins = builder.fp_den(&x3_numerator, &d_mul_f, true);

        // y3 = y3_numerator / (1 - d * f).
        let y3_ins = builder.fp_den(&y3_numerator, &d_mul_f, false);

        // R = (x3, y3).
        AffinePointRegister::new(x3_ins.result, y3_ins.result)
    }

    fn ec_double(
        builder: &mut AirBuilder<L>,
        p: &AffinePointRegister<Self>,
    ) -> AffinePointRegister<Self> {
        Self::ec_add(builder, p, p)
    }
}
