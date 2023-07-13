use plonky2::field::types::{
    Field as Plonky2Field, PrimeField as Plonky2PrimeField, Sample as Plonky2Sample,
};

pub use crate::math::prelude::*;

impl<F: Plonky2Field> Ring for F {
    const ONE: Self = F::ONE;
    const ZERO: Self = F::ZERO;
}

impl<F: Plonky2Field> Field for F {
    fn try_inverse(&self) -> Option<Self> {
        Some(self.inverse())
    }
    fn from_canonical_u8(n: u8) -> Self {
        F::from_canonical_u8(n)
    }
    fn from_canonical_u16(n: u16) -> Self {
        F::from_canonical_u16(n)
    }
    fn from_canonical_u32(n: u32) -> Self {
        F::from_canonical_u32(n)
    }
    fn from_canonical_u64(n: u64) -> Self {
        F::from_canonical_u64(n)
    }
    fn from_canonical_usize(n: usize) -> Self {
        F::from_canonical_usize(n)
    }

    fn from_noncanonical_biguint(n: num::BigUint) -> Self {
        F::from_noncanonical_biguint(n)
    }
}

impl<F: Plonky2Sample> Sample for F {
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        F::sample(rng)
    }
}

impl<F: Plonky2PrimeField> PrimeField for F {

}
