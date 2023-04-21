pub mod gadget;
pub mod ops;

use core::fmt::Debug;
use core::iter;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

pub use gadget::PolynomialGadget;
use itertools::Itertools;
use num::BigUint;
pub use ops::PolynomialOps;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

use super::field::FieldParameters;
use crate::arithmetic::utils::{bigint_into_u16_digits, biguint_to_16_digits_field};

/// A wrapper around a vector of field elements that implements polynomial operations. Often used
/// to represent polynomials when computing with traces.
#[derive(Debug, Clone)]
pub struct Polynomial<F> {
    pub coefficients: Vec<F>,
}

impl<T: Clone> Polynomial<T> {
    pub fn new_from_vec(coefficients: Vec<T>) -> Self {
        Self { coefficients }
    }

    pub fn from_biguint_field(num: &BigUint, num_bits: usize, num_limbs: usize) -> Self
    where
        T: Field,
    {
        assert_eq!(num_bits, 16, "Only 16 bit numbers supported");
        Self::new_from_vec(biguint_to_16_digits_field(num, num_limbs))
    }

    pub fn from_biguint<P: FieldParameters>(num: &BigUint) -> Self
    where
        T: From<u16>,
    {
        assert_eq!(P::NB_BITS_PER_LIMB, 16, "Only 16 bit numbers supported");
        let num_limbs = bigint_into_u16_digits(num, P::NB_LIMBS)
            .iter()
            .map(|x| (*x).into())
            .collect();
        Self::new_from_vec(num_limbs)
    }

    pub fn new_from_slice(coefficients: &[T]) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
        }
    }

    pub fn from_slice(coefficients: &[T]) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
        }
    }

    pub fn from_polynomial<S>(p: Polynomial<S>) -> Self
    where
        S: Into<T>,
    {
        Self {
            coefficients: p.coefficients.into_iter().map(|x| x.into()).collect(),
        }
    }

    pub fn into_vec(self) -> Vec<T> {
        self.coefficients
    }

    pub fn coefficients(&self) -> Vec<T> {
        self.coefficients.clone()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.coefficients
    }

    pub fn as_mut_vec(&mut self) -> &mut Vec<T> {
        &mut self.coefficients
    }

    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    pub fn constant(value: T) -> Self {
        Self {
            coefficients: vec![value],
        }
    }
}

impl<T> FromIterator<T> for Polynomial<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            coefficients: iter.into_iter().collect(),
        }
    }
}

impl<T: Default + Clone> Polynomial<T> {
    pub fn constant_monomial(c: T, degree: usize) -> Self {
        let mut coefficients = vec![T::default(); degree + 1];
        coefficients[degree] = c;
        Self { coefficients }
    }
}

impl<T> Polynomial<T> {
    pub fn eval<S>(&self, x: S) -> S
    where
        T: Copy,
        S: One<Output = S>,
        S: Add<Output = S> + MulAssign + Mul<T, Output = S> + Copy + iter::Sum,
    {
        PolynomialOps::eval::<T, S, S>(self.as_slice(), &x)
    }

    pub fn root_quotient(&self, r: T) -> Self
    where
        T: Copy
            + Neg<Output = T>
            + Debug
            + MulAssign
            + Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + PartialEq
            + Eq
            + iter::Sum,
    {
        Self::new_from_vec(PolynomialOps::root_quotient(self.as_slice(), &r))
    }
}

impl<F: Field> Polynomial<F> {
    pub fn eval_field(&self, x: F) -> F {
        PolynomialOps::eval::<F, Eval<F>, F>(self.as_slice(), &x)
    }

    pub fn from_canonical_u64(p: Polynomial<u64>) -> Self {
        Self::new_from_vec(
            p.coefficients
                .into_iter()
                .map(|x| F::from_canonical_u64(x))
                .collect(),
        )
    }

    pub fn x() -> Self {
        Self::new_from_vec(vec![F::ZERO, F::ONE])
    }

    pub fn x_n(n: usize) -> Self {
        let mut result = vec![F::ZERO; n + 1];
        result[n] = F::ONE;
        Self::new_from_vec(result)
    }
}

impl<T: Add<Output = T> + Copy + Default> Add for Polynomial<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new_from_vec(PolynomialOps::add(self.as_slice(), other.as_slice()))
    }
}

impl<T: Add<Output = T> + Copy + Default> Add for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn add(self, other: Self) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::add(self.as_slice(), other.as_slice()))
    }
}

impl<T: Add<Output = T> + Copy + Default> Add<&Polynomial<T>> for Polynomial<T> {
    type Output = Polynomial<T>;

    fn add(self, other: &Polynomial<T>) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::add(self.as_slice(), other.as_slice()))
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + AddAssign + Copy + Default> Add<T> for Polynomial<T> {
    type Output = Polynomial<T>;

    fn add(self, other: T) -> Polynomial<T> {
        let mut coefficients = self.coefficients();
        coefficients[0] += other;
        Self::new_from_vec(coefficients)
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + AddAssign + Copy + Default> Add<T> for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn add(self, other: T) -> Polynomial<T> {
        let mut coefficients = self.coefficients();
        coefficients[0] += other;
        Polynomial::new_from_vec(coefficients)
    }
}

impl<T: Neg<Output = T> + Copy> Neg for Polynomial<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new_from_vec(PolynomialOps::neg(self.as_slice()))
    }
}

impl<T: Sub<Output = T> + Neg<Output = T> + Copy + Default> Sub for Polynomial<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new_from_vec(PolynomialOps::sub(self.as_slice(), other.as_slice()))
    }
}

impl<T: Sub<Output = T> + Neg<Output = T> + Copy + Default> Sub<&Polynomial<T>> for Polynomial<T> {
    type Output = Polynomial<T>;

    fn sub(self, other: &Polynomial<T>) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::sub(self.as_slice(), other.as_slice()))
    }
}

impl<T: Sub<Output = T> + Neg<Output = T> + Copy + Default> Sub for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn sub(self, other: Self) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::sub(self.as_slice(), other.as_slice()))
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + Copy + Default> Mul for Polynomial<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new_from_vec(PolynomialOps::mul(self.as_slice(), other.as_slice()))
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + Copy + Default> Mul for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn mul(self, other: Self) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::mul(self.as_slice(), other.as_slice()))
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + Copy + Default> Mul<T> for Polynomial<T> {
    type Output = Self;

    fn mul(self, other: T) -> Self {
        Self::new_from_vec(PolynomialOps::scalar_mul(self.as_slice(), &other))
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + Copy + Default> Mul<T> for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn mul(self, other: T) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::scalar_mul(self.as_slice(), &other))
    }
}

impl<T: Default + Clone> Default for Polynomial<T> {
    fn default() -> Self {
        Self::new_from_vec(vec![T::default()])
    }
}

pub struct Eval<T>(T);

impl<T> From<T> for Eval<T> {
    fn from(f: T) -> Self {
        Self(f)
    }
}

pub struct PowersIter<T> {
    base: T,
    current: T,
}

impl<T: MulAssign + Copy> Iterator for PowersIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let result = self.current;
        self.current *= self.base;
        Some(result)
    }
}

pub trait One {
    type Output;
    fn one() -> Self::Output;
}

pub trait Zero {
    type Output;
    fn zero() -> Self::Output;
}

impl<F: Field> One for Eval<F> {
    type Output = F;

    fn one() -> Self::Output {
        F::ONE
    }
}

impl<F: Field> Zero for Eval<F> {
    type Output = F;

    fn zero() -> Self::Output {
        F::ZERO
    }
}

pub fn get_powers<T>(x: T, one: T) -> PowersIter<T> {
    PowersIter {
        base: x,
        current: one,
    }
}

pub fn to_u16_le_limbs_polynomial<F: RichField, P: FieldParameters>(x: &BigUint) -> Polynomial<F> {
    let num_limbs = bigint_into_u16_digits(x, P::NB_LIMBS)
        .iter()
        .map(|x| F::from_canonical_u16(*x))
        .collect();
    Polynomial::new_from_vec(num_limbs)
}
