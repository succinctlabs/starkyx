use num::{BigUint, Zero};
use serde::{Deserialize, Serialize};

use super::point::{AffinePoint, AffinePointRegister};
use super::{EllipticCurve, EllipticCurveAir, EllipticCurveParameters};
use crate::chip::builder::AirBuilder;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::parameters::{FieldParameters, MAX_NB_LIMBS};
use crate::chip::AirParameters;

pub mod biguint_operations;
pub mod bn254;
pub mod group;
pub mod slope;

/// Parameters that specify a short Weierstrass curve : y^2 = x^3 + ax + b.
pub trait WeierstrassParameters: EllipticCurveParameters {
    const A: [u16; MAX_NB_LIMBS];
    const B: [u16; MAX_NB_LIMBS];

    fn generator() -> (BigUint, BigUint);

    fn prime_group_order() -> BigUint;

    fn a_int() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::A.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }

    fn b_int() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::B.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }

    fn nb_scalar_bits() -> usize {
        Self::BaseField::NB_LIMBS * 16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SWCurve<E>(pub E);

impl<E: WeierstrassParameters> EllipticCurveParameters for SWCurve<E> {
    type BaseField = E::BaseField;
}

impl<E: WeierstrassParameters> EllipticCurve for SWCurve<E> {
    fn ec_add(p: &AffinePoint<Self>, q: &AffinePoint<Self>) -> AffinePoint<Self> {
        p.sw_add(q)
    }

    fn ec_double(p: &AffinePoint<Self>) -> AffinePoint<Self> {
        p.sw_double()
    }

    fn ec_generator() -> AffinePoint<Self> {
        let (x, y) = E::generator();
        AffinePoint::new(x, y)
    }

    fn ec_neutral() -> Option<AffinePoint<Self>> {
        None
    }

    fn ec_neg(p: &AffinePoint<Self>) -> AffinePoint<Self> {
        let modulus = E::BaseField::modulus();
        AffinePoint::new(p.x.clone(), modulus - &p.y)
    }
}

impl<E: WeierstrassParameters> SWCurve<E> {
    pub fn generator() -> AffinePoint<SWCurve<E>> {
        let (x, y) = E::generator();

        AffinePoint::new(x, y)
    }

    pub fn a_int() -> BigUint {
        E::a_int()
    }

    pub fn b_int() -> BigUint {
        E::b_int()
    }
}

impl<L: AirParameters, E: WeierstrassParameters> EllipticCurveAir<L> for SWCurve<E>
where
    L::Instruction: FromFieldInstruction<E::BaseField>,
{
    fn ec_add_air(
        builder: &mut AirBuilder<L>,
        p: &AffinePointRegister<Self>,
        q: &AffinePointRegister<Self>,
    ) -> AffinePointRegister<Self> {
        builder.sw_add::<E>(p, q)
    }

    fn ec_double_air(
        builder: &mut AirBuilder<L>,
        p: &AffinePointRegister<Self>,
    ) -> AffinePointRegister<Self> {
        // TODO: might be expensive for no reason if doing more than one add in a row.
        // otherwise, there is no extra cost.
        let a = builder.fp_constant(&E::a_int());
        let three = builder.fp_constant(&BigUint::from(3u32));

        builder.sw_double::<E>(p, &a, &three)
    }

    fn ec_generator_air(builder: &mut AirBuilder<L>) -> AffinePointRegister<Self> {
        let (x, y) = E::generator();

        let x = builder.fp_constant(&x);
        let y = builder.fp_constant(&y);

        AffinePointRegister::new(x, y)
    }
}
