use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};

use crate::math::extension::cubic::element::CubicElement;
use crate::math::extension::cubic::extension::CubicExtension;
use crate::math::extension::cubic::parameters::CubicParameters;

pub type GF3 = CubicExtension<GoldilocksField, GoldilocksCubicParameters>;

/// Galois parameters for the cubic Goldilocks extension field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldilocksCubicParameters;

impl CubicParameters<GoldilocksField> for GoldilocksCubicParameters {
    const GALOIS_ORBIT: [CubicElement<GoldilocksField>; 2] = [
        CubicElement([
            GoldilocksField(7831040667286096068),
            GoldilocksField(10050274602728160328),
            GoldilocksField(6700183068485440219),
        ]),
        CubicElement([
            GoldilocksField(10615703402128488253),
            GoldilocksField(8396469466686423992),
            GoldilocksField(11746561000929144102),
        ]),
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::prelude::*;

    #[test]
    fn test_gf3_add() {
        let num_tests = 100;

        for _ in 0..num_tests {
            let a = GF3::rand();
            let b = GF3::rand();

            let a_rr = a.0.as_array();
            let b_rr = b.0.as_array();

            assert_eq!(a + b, b + a);
            assert_eq!(a, a + GF3::ZERO);
            assert_eq!(
                (a + b).0.as_array(),
                [a_rr[0] + b_rr[0], a_rr[1] + b_rr[1], a_rr[2] + b_rr[2]]
            );
        }
    }

    #[test]
    fn test_gf3_mul() {
        let num_tests = 100;

        for _ in 0..num_tests {
            let a = GF3::rand();
            let b = GF3::rand();
            let c = GF3::rand();

            assert_eq!(a * b, b * a);
            assert_eq!(a * (b * c), (a * b) * c);
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!(a * GF3::ONE, a);
            assert_eq!(a * GF3::ZERO, GF3::ZERO);
        }
    }

    #[test]
    fn test_orbit() {
        for &g in GF3::ORBIT.iter() {
            assert_eq!(g * g * g, g - GF3::ONE);
        }
    }

    #[test]
    fn test_gf3_inverse() {
        let num_tests = 100;

        for _ in 0..num_tests {
            let a = GF3::rand();

            let a_inv = a.inverse();

            assert_eq!(a * a_inv, GF3::ONE);
        }
    }
}
