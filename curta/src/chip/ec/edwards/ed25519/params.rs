use num::{BigUint, Num, One};
use serde::{Deserialize, Serialize};

use crate::chip::ec::edwards::{EdwardsCurve, EdwardsParameters};
use crate::chip::ec::EllipticCurveParameters;
use crate::chip::field::parameters::{FieldParameters, MAX_NB_LIMBS};

pub type Ed25519 = EdwardsCurve<Ed25519Parameters>;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ed25519Parameters;

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ed25519BaseField;

impl FieldParameters for Ed25519BaseField {
    const NB_BITS_PER_LIMB: usize = 16;
    const NB_LIMBS: usize = 16;
    const NB_WITNESS_LIMBS: usize = 2 * Self::NB_LIMBS - 2;
    const MODULUS: [u16; MAX_NB_LIMBS] = [
        65517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
        65535, 65535, 32767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const WITNESS_OFFSET: usize = 1usize << 21;

    fn modulus() -> BigUint {
        (BigUint::one() << 255) - BigUint::from(19u32)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ed25519ScalarField;

impl FieldParameters for Ed25519ScalarField {
    const NB_BITS_PER_LIMB: usize = 16;
    const NB_LIMBS: usize = 16;
    const NB_WITNESS_LIMBS: usize = 2 * Self::NB_LIMBS - 2;
    const MODULUS: [u16; MAX_NB_LIMBS] = [
        54253, 23797, 25370, 22546, 40150, 41719, 63966, 5342, 0, 0, 0, 0, 0, 0, 0, 4096, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const WITNESS_OFFSET: usize = 1usize << 21;

    fn modulus() -> BigUint {
        (BigUint::one() << 252) + BigUint::from(27742317777372353535851937790883648493u128)
    }
}

impl EllipticCurveParameters for Ed25519Parameters {
    type BaseField = Ed25519BaseField;
}

impl EdwardsParameters for Ed25519Parameters {
    const D: [u16; MAX_NB_LIMBS] = [
        30883, 4953, 19914, 30187, 55467, 16705, 2637, 112, 59544, 30585, 16505, 36039, 65139,
        11119, 27886, 20995, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    fn prime_group_order() -> BigUint {
        BigUint::from(2u32).pow(252) + BigUint::from(27742317777372353535851937790883648493u128)
    }

    fn generator() -> (BigUint, BigUint) {
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
        (x, y)
    }
}
