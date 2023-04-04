use num::{BigUint, Num, One};

pub mod denominator;
pub mod ec_add;
pub mod quad;

pub const LIMB: u32 = 2u32.pow(16);

pub const P: [u16; 16] = [
    65517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
    65535, 65535, 32767,
];

pub const D: [u16; 16] = [
    30883, 4953, 19914, 30187, 55467, 16705, 2637, 112, 59544, 30585, 16505, 36039, 65139, 11119,
    27886, 20995,
];

pub fn get_p() -> BigUint {
    (BigUint::one() << 255) - BigUint::from(19u32)
}

pub fn get_d() -> BigUint {
    BigUint::from_str_radix(
        "37095705934669439343138083508754565189542113879843219016388785533085940283555",
        10,
    )
    .unwrap()
}

/// Layoutds for the Opcodes that comprise any Edwards curve operation.
#[derive(Debug, Clone, Copy)]
pub enum EdOpcodeLayout {
    Quad(quad::QuadLayout),
}

/// The core Opcodes that comprise any Edwards curve operation.
#[derive(Debug, Clone)]
pub enum EdOpcode {
    /// Quad(x_1, x_2, x_3, x_4) = (x_1 * x_2 + x_3 * x_4) mod p
    Quad(BigUint, BigUint, BigUint, BigUint),

    /// DEN(x_1, x_2, y_1, y_2, sign) = 1 + sign * D * (x_1 * x_2 * y_1 * y_2) mod p
    DEN(BigUint, BigUint, BigUint, BigUint, bool),
}

#[cfg(test)]
mod tests {

    use num::{BigUint, Num};

    use super::*;
    use crate::arithmetic::util;

    #[test]
    fn check_p25519_value() {
        let p = BigUint::from(2u32).pow(255) - BigUint::from(19u32);
        let p_limbs = util::bigint_into_u16_digits(&p, 16);

        assert_eq!(p_limbs, P);
        assert_eq!(p, get_p());
    }

    #[test]
    fn check_d_value() {
        let d = BigUint::from_str_radix(
            "37095705934669439343138083508754565189542113879843219016388785533085940283555",
            10,
        )
        .unwrap();

        // check the value of d is correct
        let p = BigUint::from(2u32).pow(255) - BigUint::from(19u32);
        assert_eq!((121666u32 * &d + 121665u32) % &p, BigUint::from(0u32));
        let d_limbs = util::bigint_into_u16_digits(&d, 16);
        assert_eq!(d_limbs, D);

        let d_from_limbs = util::digits_to_biguint(&D);
        assert_eq!(d, d_from_limbs);
        assert_eq!(d, get_d());
    }
}
