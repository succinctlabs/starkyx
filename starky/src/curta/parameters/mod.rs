use num::{BigUint, Zero};

use super::ec::affine::AffinePoint;

pub mod ed25519;

pub const MAX_NB_LIMBS: usize = 32;
pub const LIMB: u32 = 2u32.pow(16);

pub trait FieldParameters: Send + Sync + Copy + 'static {
    const NB_BITS_PER_LIMB: usize;
    const NB_LIMBS: usize;
    const NB_WITNESS_LIMBS: usize;
    const MODULUS: [u16; MAX_NB_LIMBS];
    const WITNESS_OFFSET: usize;

    fn modulus() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::MODULUS.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }
}

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
