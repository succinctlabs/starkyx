use core::fmt::Debug;
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::BigUint;
use rand::rngs::OsRng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// A trait for an Abstract ring containing addition, multiplication, and a zero element
pub trait Ring:
    Debug
    + Sized
    + Default
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + SubAssign
    + Neg<Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + Sum
    + Product
{
    const ONE: Self;
    const ZERO: Self;

    #[must_use]
    fn square(&self) -> Self {
        self.clone() * self.clone()
    }
}

#[derive(Clone)]
pub struct Powers<R: Ring> {
    base: R,
    current: R,
}

impl<R: Ring + Copy> Iterator for Powers<R> {
    type Item = R;

    fn next(&mut self) -> Option<R> {
        let result = self.current;
        self.current *= self.base;
        Some(result)
    }
}

/// The basic trait for a finite field
pub trait Field:
    Ring
    + Div<Output = Self>
    + DivAssign
    + 'static
    + Copy
    + Eq
    + Hash
    + Send
    + Sync
    + Serialize
    + DeserializeOwned
{
    /// Inverts `self`, returning `None` if `self` is zero.
    fn try_inverse(&self) -> Option<Self>;

    fn from_canonical_u8(n: u8) -> Self;
    fn from_canonical_u16(n: u16) -> Self;
    fn from_canonical_u32(n: u32) -> Self;
    fn from_canonical_u64(n: u64) -> Self;
    fn from_canonical_usize(n: usize) -> Self;
    fn from_noncanonical_biguint(n: BigUint) -> Self;

    fn primitive_root_of_unity(n_log: usize) -> Self;

    fn two_adic_subgroup(n_log: usize) -> Vec<Self>;

    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

    /// Returns `true` if `self` is zero.
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    /// Raise `self` to the power of `power`.
    fn pow(&self, power: u64) -> Self {
        let mut current = *self;
        let mut product = Self::ONE;

        let n_bits = 64 - power.leading_zeros();
        for j in 0..n_bits {
            if (power >> j & 1) != 0 {
                product *= current;
            }
            current = current.square();
        }
        product
    }

    fn pow_biguint(&self, power: &BigUint) -> Self {
        let mut result = Self::ONE;
        for &digit in power.to_u64_digits().iter().rev() {
            result = result.two_pow(64);
            result *= self.pow(digit);
        }
        result
    }

    /// Compute `self^{2^{power_log}}` in the field.
    fn two_pow(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    /// An iterator over the powers of `self`.
    ///
    /// The iterator starts with `Self::ONE` and never terminates.
    fn powers(&self) -> Powers<Self> {
        Powers {
            base: *self,
            current: Self::ONE,
        }
    }
}

/// A finite field of the form `F_p` for some prime `p`.
pub trait PrimeField: Field {}

/// A prime field of order less than `2^64`.
pub trait PrimeField64: PrimeField + Serialize + for<'de> Deserialize<'de> {
    // const ORDER_U64: u64;

    fn as_canonical_u64(&self) -> u64;

    fn order() -> u64 {
        (-Self::ONE).as_canonical_u64() + 1
    }
}

/// A prime field of order less than `2^32`.
pub trait PrimeField32: PrimeField {
    // const ORDER_U32: u32;

    fn as_canonical_u32(&self) -> u32;
}

// default impl<F: PrimeField32> PrimeField64 for F {
//     // default const ORDER_U64: u64 = <F as PrimeField32>::ORDER_U32 as u64;

//     default fn as_canonical_u64(&self) -> u64 {
//         u64::from(self.as_canonical_u32())
//     }
// }

/// Sampling of a random value.
pub trait Sample: Sized {
    /// Samples a single value using `rng`.
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized;

    /// Samples a single value using the [`OsRng`].
    #[inline]
    fn rand() -> Self {
        Self::sample(&mut OsRng)
    }

    /// Samples a [`Vec`] of values of length `n` using [`OsRng`].
    #[inline]
    fn rand_vec(n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::rand()).collect()
    }

    /// Samples an array of values of length `N` using [`OsRng`].
    #[inline]
    fn rand_array<const N: usize>() -> [Self; N] {
        Self::rand_vec(N)
            .try_into()
            .ok()
            .expect("This conversion can never fail.")
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn ring_test<F: Ring + Eq + Sample>() {
        let a = F::rand();
        let b = F::rand();
        let c = F::rand();

        let one = F::ONE;
        let zero = F::ZERO;

        // Test that a + 0 = a
        assert_eq!(a.clone() + zero.clone(), a);
        // Test that a * 1 = a
        assert_eq!(a.clone() * one.clone(), a);
        // Test additive commutativity
        assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());
        // Test additive associativity
        assert_eq!(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone())
        );
        // Test additive identity
        assert_eq!(a.clone() + (-a.clone()), zero);
        // Test multiplicative commutativity
        assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());
        // Test multiplicative associativity
        assert_eq!(
            (a.clone() * b.clone()) * c.clone(),
            a.clone() * (b.clone() * c.clone())
        );
        // Test multiplicative identity
        assert_eq!(a.clone() * one, a);
        // Test distributivity
        assert_eq!(a.clone() * (b.clone() + c.clone()), a.clone() * b + a * c);
    }

    pub fn field_test<F: Field + Sample>() {
        let a = F::rand();
        let b = F::rand();
        let c = F::rand();

        let one = F::ONE;
        let zero = F::ZERO;

        // Test that a + 0 = a
        assert_eq!(a + zero, a);
        // Test that a * 1 = a
        assert_eq!(a * one, a);
        // Test additive commutativity
        assert_eq!(a.add(b), b.add(a));
        // Test additive associativity
        assert_eq!((a + b) + c, a + (b + c));
        // Test additive identity
        assert_eq!(a + (-a), zero);
        // Test multiplicative commutativity
        assert_eq!(a * b, b * a);
        // Test multiplicative associativity
        assert_eq!((a * b) * c, a * (b * c));
        // Test multiplicative identity
        assert_eq!(a * one, a);
        // Test distributivity
        assert_eq!(a * (b + c), a * b + a * c);

        // Test multiplicative inverse
        if a != zero {
            assert_eq!(a * a.inverse(), one);
        }
    }
}
