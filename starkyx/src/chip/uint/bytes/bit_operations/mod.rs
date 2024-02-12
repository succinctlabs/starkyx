pub use crate::math::prelude::*;

pub mod and;
pub mod not;
pub mod rotate;
pub mod shift;
pub mod xor;

pub mod util {

    #[inline]
    pub fn u8_to_bits_le(x: u8) -> [u8; 8] {
        core::array::from_fn(|i| (x >> i) & 1)
    }

    #[inline]
    pub fn bits_u8_to_val(bits: &[u8]) -> u8 {
        bits.iter().enumerate().map(|(i, b)| b << i).sum::<u8>()
    }
}
