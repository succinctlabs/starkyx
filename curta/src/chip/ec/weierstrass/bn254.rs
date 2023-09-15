//! Parameters for bn254 base field.
use num::BigUint;
use serde::{Deserialize, Serialize};

use super::projective::SWProjectivePoint;
use super::WeierstrassParameter;
use crate::chip::ec::EllipticCurveParameters;
use crate::chip::field::parameters::FieldParameters;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Bn254 curve parameter
pub struct Bn254;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

impl EllipticCurveParameters for Bn254 {
    type BaseField = Bn254BaseField;
}

impl WeierstrassParameter for Bn254 {
    const A: [u16; crate::chip::field::parameters::MAX_NB_LIMBS] = [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];

    fn generator() -> SWProjectivePoint<Self> {
        let x = BigUint::from(1u32);
        let y = BigUint::from(2u32);
        let z = BigUint::from(1u32);
        SWProjectivePoint::new(x, y, z)
    }

    fn prime_group_order() -> num::BigUint {
        todo!()
    }
}
