//! Edwards curves

pub mod den;

use num::Zero;

use super::*;
use crate::arithmetic::field::Fp25519Param;

pub trait EdwardsParameters<const N_LIMBS: usize>: EllipticCurveParameters<N_LIMBS> {
    const D: [u16; N_LIMBS];

    fn d_biguint() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::D.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ed25519Parameters;

impl EllipticCurveParameters<16> for Ed25519Parameters {
    type FieldParam = Fp25519Param;
}

impl EdwardsParameters<16> for Ed25519Parameters {
    const D: [u16; 16] = [
        30883, 4953, 19914, 30187, 55467, 16705, 2637, 112, 59544, 30585, 16505, 36039, 65139,
        11119, 27886, 20995,
    ];
}
