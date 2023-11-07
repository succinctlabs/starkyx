use num::{BigUint, Num, Zero};
use serde::{Deserialize, Serialize};

use super::{SWCurve, WeierstrassParameters};
use crate::chip::ec::EllipticCurveParameters;
use crate::chip::field::parameters::FieldParameters;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// Bn254 curve parameter
pub struct Bn254Parameters;

pub type Bn254 = SWCurve<Bn254Parameters>;

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// Bn254 base field parameter
pub struct Bn254BaseField;

impl FieldParameters for Bn254BaseField {
    const NB_BITS_PER_LIMB: usize = 16;

    const NB_LIMBS: usize = 16;

    const NB_WITNESS_LIMBS: usize = 2 * Self::NB_LIMBS - 2;

    // Base field modulus:
    //  21888242871839275222246405745257275088696311157297823662689037894645226208583
    const MODULUS: [u16; crate::chip::field::parameters::MAX_NB_LIMBS] = [
        64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001,
        57649, 20082, 12388, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    const WITNESS_OFFSET: usize = 1usize << 20;
}

impl EllipticCurveParameters for Bn254Parameters {
    type BaseField = Bn254BaseField;
}

impl WeierstrassParameters for Bn254Parameters {
    const A: [u16; crate::chip::field::parameters::MAX_NB_LIMBS] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];

    const B: [u16; crate::chip::field::parameters::MAX_NB_LIMBS] = [
        3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    fn generator() -> (BigUint, BigUint) {
        let x = BigUint::from(1u32);
        let y = BigUint::from(2u32);
        (x, y)
    }

    fn prime_group_order() -> num::BigUint {
        BigUint::from_str_radix(
            "21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap()
    }

    fn a_int() -> BigUint {
        BigUint::zero()
    }

    fn b_int() -> BigUint {
        BigUint::from(3u32)
    }
}
