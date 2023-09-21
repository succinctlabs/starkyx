//! Gadgets for Weierstrass elliptic curve

use num::{BigUint, Zero};

use self::projective::SWProjectivePoint;
use crate::chip::ec::EllipticCurveParameters;
use crate::chip::field::parameters::{FieldParameters, MAX_NB_LIMBS};

pub mod add;
pub mod bn254;
pub mod projective;

/// Parameters for Weierstrass elliptic curve of the form
/// y^2 = x^3 + a * x + b
pub trait WeierstrassParameter: EllipticCurveParameters {
    /// Constant `a` of the Weierstrass curve
    const A: [u16; MAX_NB_LIMBS];

    /// Constant `b` of the Weierstrass curve
    const B: [u16; MAX_NB_LIMBS];

    /// Returns a generator
    fn generator() -> SWProjectivePoint<Self>;

    /// Primal group order of the curve
    fn prime_group_order() -> BigUint;

    /// Returns the constant `a`
    fn a_biguint() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::A.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }

    /// Returns the constant `b`
    fn b_biguint() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::B.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }

    /// Number of bits for scalar
    fn nb_scalar_bits() -> usize {
        Self::BaseField::NB_LIMBS * 16
    }

    /// Returns a neutral point
    fn neutral() -> SWProjectivePoint<Self> {
        SWProjectivePoint::new(
            BigUint::from(1u32),
            BigUint::from(1u32),
            BigUint::from(0u32),
        )
    }
}
