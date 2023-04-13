//! Edwards curves

pub mod add;
pub mod bigint_operations;
pub mod den;
pub mod instructions;
pub mod scalar_mul;

use num::{Num, Zero};

use super::*;
use crate::arithmetic::field::{Fp25519Param, MAX_NB_LIMBS};

pub trait EdwardsParameters: EllipticCurveParameters {
    const D: [u16; MAX_NB_LIMBS];

    /// Returns the canonical generator
    fn generator() -> AffinePoint<Self>;

    fn prime_group_order() -> BigUint;

    fn d_biguint() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::D.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }

    fn num_scalar_bits() -> usize {
        Self::FieldParam::NB_LIMBS * 16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ed25519Parameters;

impl EllipticCurveParameters for Ed25519Parameters {
    type FieldParam = Fp25519Param;
}

impl EdwardsParameters for Ed25519Parameters {
    const D: [u16; MAX_NB_LIMBS] = [
        30883, 4953, 19914, 30187, 55467, 16705, 2637, 112, 59544, 30585, 16505, 36039, 65139,
        11119, 27886, 20995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    fn prime_group_order() -> BigUint {
        BigUint::from(2u32).pow(252) + BigUint::from(27742317777372353535851937790883648493u128)
    }

    fn generator() -> AffinePoint<Self> {
        let x = BigUint::from_str_radix(
            "15112221349535400772501151409588531511454012693041857206046113283949847762202",
            10,
        )
        .unwrap();
        let y = BigUint::from_str_radix(
            "46316835694926478169428394003475163141307993866256225615783033603165251855960",
            10,
        )
        .unwrap();
        AffinePoint::new(x, y)
    }
}
