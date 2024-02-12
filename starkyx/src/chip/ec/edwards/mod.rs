use num::{BigUint, Zero};
use serde::{Deserialize, Serialize};

use super::point::{AffinePoint, AffinePointRegister};
use super::{EllipticCurve, EllipticCurveAir, EllipticCurveParameters};
use crate::chip::builder::AirBuilder;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::parameters::{FieldParameters, MAX_NB_LIMBS};
use crate::chip::AirParameters;
pub mod add;
pub mod assert_valid;
pub mod bigint_operations;
pub mod ed25519;

pub trait EdwardsParameters: EllipticCurveParameters {
    const D: [u16; MAX_NB_LIMBS];

    fn generator() -> (BigUint, BigUint);

    fn prime_group_order() -> BigUint;

    fn d_biguint() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::D.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }

    fn neutral() -> (BigUint, BigUint) {
        (BigUint::from(0u32), BigUint::from(1u32))
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EdwardsCurve<E: EdwardsParameters>(pub E);

impl<E: EdwardsParameters> EllipticCurveParameters for EdwardsCurve<E> {
    type BaseField = E::BaseField;
}

impl<E: EdwardsParameters> EdwardsCurve<E> {
    pub fn prime_group_order() -> BigUint {
        E::prime_group_order()
    }

    pub fn neutral() -> AffinePoint<Self> {
        let (x, y) = E::neutral();
        AffinePoint::new(x, y)
    }
}

impl<E: EdwardsParameters> EllipticCurve for EdwardsCurve<E> {
    fn ec_add(p: &AffinePoint<Self>, q: &AffinePoint<Self>) -> AffinePoint<Self> {
        p.ed_add(q)
    }

    fn ec_double(p: &AffinePoint<Self>) -> AffinePoint<Self> {
        p.ed_double()
    }

    fn ec_generator() -> AffinePoint<Self> {
        let (x, y) = E::generator();
        AffinePoint::new(x, y)
    }

    fn ec_neutral() -> Option<AffinePoint<Self>> {
        Some(Self::neutral())
    }

    fn ec_neg(p: &AffinePoint<Self>) -> AffinePoint<Self> {
        let modulus = E::BaseField::modulus();
        AffinePoint::new(&modulus - &p.x, p.y.clone())
    }
}

impl<L: AirParameters, E: EdwardsParameters> EllipticCurveAir<L> for EdwardsCurve<E>
where
    L::Instruction: FromFieldInstruction<E::BaseField>,
{
    fn ec_add_air(
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
        let x3_numerator = builder.fp_inner_product(&[x1, x2], &[y2, y1]);

        // y3_numerator = y1 * y2 + x1 * x2.
        let y3_numerator = builder.fp_inner_product(&[y1, x1], &[y2, x2]);

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

    fn ec_double_air(
        builder: &mut AirBuilder<L>,
        p: &AffinePointRegister<Self>,
    ) -> AffinePointRegister<Self> {
        Self::ec_add_air(builder, p, p)
    }

    fn ec_generator_air(builder: &mut AirBuilder<L>) -> AffinePointRegister<Self> {
        let (x_int, y_int) = E::generator();

        let x = builder.fp_constant(&x_int);
        let y = builder.fp_constant(&y_int);

        AffinePointRegister::new(x, y)
    }
}
