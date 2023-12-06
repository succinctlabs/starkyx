use core::fmt::Debug;

use num::bigint::RandBigInt;
use num::{BigUint, Zero};
use rand::rngs::OsRng;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub const MAX_NB_LIMBS: usize = 32;
pub const LIMB: u32 = 2u32.pow(16);

pub trait FieldParameters:
    Send + Sync + Copy + 'static + Debug + Serialize + DeserializeOwned + Default
{
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

    fn nb_bits() -> usize {
        Self::NB_BITS_PER_LIMB * Self::NB_LIMBS
    }

    fn rand() -> BigUint {
        OsRng.gen_biguint_below(&Self::modulus())
    }
}

#[cfg(test)]
pub mod tests {
    use num::One;
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Fp25519;

    impl FieldParameters for Fp25519 {
        const NB_BITS_PER_LIMB: usize = 16;
        const NB_LIMBS: usize = 16;
        const NB_WITNESS_LIMBS: usize = 2 * Self::NB_LIMBS - 2;
        const MODULUS: [u16; MAX_NB_LIMBS] = [
            65517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
            65535, 65535, 65535, 32767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        const WITNESS_OFFSET: usize = 1usize << 21;

        fn modulus() -> BigUint {
            (BigUint::one() << 255) - BigUint::from(19u32)
        }
    }
}
