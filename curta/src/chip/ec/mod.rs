use num::{BigUint, Zero};

use self::point::AffinePoint;
use super::field::parameters::{FieldParameters, MAX_NB_LIMBS};

pub mod gadget;
pub mod point;

pub trait EllipticCurveParameters: Send + Sync + Copy + 'static {
    type BaseField: FieldParameters;
}

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
