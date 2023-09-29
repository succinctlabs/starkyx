use num::{BigUint, Zero};

use super::point::AffinePoint;
use super::EllipticCurveParameters;
use crate::chip::field::parameters::{FieldParameters, MAX_NB_LIMBS};

pub mod biguint_operations;
pub mod bn254;
pub mod group;
pub mod slope;

/// Parameters that specify a short Weierstrass curve : y^2 = x^3 + ax + b.
pub trait WeierstrassParameters: EllipticCurveParameters {
    const A: [u16; MAX_NB_LIMBS];
    const B: [u16; MAX_NB_LIMBS];

    fn generator() -> AffinePoint<Self>;

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
