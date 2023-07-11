use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::Rng;

use crate::math::prelude::*;

/// Parameters for the cubic extension F[X]/(X^3 - X - 1)
pub trait CubicParameters<F>:
    'static + Sized + Copy + Clone + Send + Sync + PartialEq + Eq + std::fmt::Debug
{
    /// The Galois orbit of the generator.
    ///
    /// These are the roots of X^3 - X - 1 in the extension field not equal to X.
    const GALOIS_ORBIT: [CubicElement<F>; 2];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubicExtension<F: Field, P: CubicParameters<F>>(pub CubicElement<F>, PhantomData<P>);

impl<F: Field, P: CubicParameters<F>> CubicExtension<F, P> {
    const ORBIT: [Self; 2] = [
        Self(P::GALOIS_ORBIT[0], PhantomData),
        Self(P::GALOIS_ORBIT[1], PhantomData),
    ];

    pub const ZERO: Self = Self::new(F::ZERO, F::ZERO, F::ZERO);
    pub const ONE: Self = Self::new(F::ONE, F::ZERO, F::ZERO);

    pub const fn new(a: F, b: F, c: F) -> Self {
        Self(CubicElement::new(a, b, c), PhantomData)
    }

    pub const fn from_base_field(a: F) -> Self {
        Self::new(a, F::ZERO, F::ZERO)
    }

    pub fn from_slice(slice: &[F]) -> Self {
        assert_eq!(slice.len(), 3);
        Self::new(slice[0], slice[1], slice[2])
    }

    #[inline]
    pub fn base_field_array(&self) -> [F; 3] {
        self.0.as_array()
    }

    #[inline]
    fn in_base_field(&self) -> bool {
        self.0.as_slice()[1] == F::ZERO && self.0.as_slice()[2] == F::ZERO
    }

    pub fn try_inverse(&self) -> Option<Self> {
        let array = self.0.as_array();
        let (a, b, c) = (array[0], array[1], array[2]);
        let gal =
            |i: usize| Self::from(a) + Self::ORBIT[i] * b + (Self::ORBIT[i] * Self::ORBIT[i]) * c;
        let (gal_1, gal_2) = (gal(0), gal(1));

        let gal_12 = gal_1 * gal_2;
        let gal_prod = *self * gal_12;
        debug_assert!(gal_prod.in_base_field());

        let gal_inv = gal_prod.0.as_slice()[0].try_inverse()?;
        Some(gal_12 * gal_inv)
    }

    pub fn inverse(&self) -> Self {
        self.try_inverse().expect("Cannot invert zero")
    }
}

impl<F: Field, P: CubicParameters<F>> From<[F; 3]> for CubicExtension<F, P> {
    fn from(value: [F; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

impl<F: Field, P: CubicParameters<F>> From<F> for CubicExtension<F, P> {
    fn from(value: F) -> Self {
        Self::from([value, F::ZERO, F::ZERO])
    }
}

impl<F: Field, P: CubicParameters<F>> Add for CubicExtension<F, P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, PhantomData)
    }
}

impl<F: Field, P: CubicParameters<F>> Add<F> for CubicExtension<F, P> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        self + Self::from_base_field(rhs)
    }
}

impl<F: Field, P: CubicParameters<F>> Sub<F> for CubicExtension<F, P> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        self - Self::from_base_field(rhs)
    }
}

impl<F: Field, P: CubicParameters<F>> Mul for CubicExtension<F, P> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0, PhantomData)
    }
}

impl<F: Field, P: CubicParameters<F>> Mul<F> for CubicExtension<F, P> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0 * rhs, PhantomData)
    }
}

impl<F: Field, P: CubicParameters<F>> Sub for CubicExtension<F, P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, PhantomData)
    }
}

impl<F: Field, P: CubicParameters<F>> Neg for CubicExtension<F, P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, PhantomData)
    }
}

impl<'a, F: Field, P: CubicParameters<F>> Sum<&'a Self> for CubicExtension<F, P> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::from([F::ZERO, F::ZERO, F::ZERO]), |acc, x| acc + *x)
    }
}

impl<F: Field, P: CubicParameters<F>> Sum for CubicExtension<F, P> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from([F::ZERO, F::ZERO, F::ZERO]), |acc, x| acc + x)
    }
}

impl<'a, F: Field, P: CubicParameters<F>> Product<&'a Self> for CubicExtension<F, P> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::from([F::ONE, F::ZERO, F::ZERO]), |acc, x| acc * *x)
    }
}

impl<F: Field, P: CubicParameters<F>> Product for CubicExtension<F, P> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from([F::ONE, F::ZERO, F::ZERO]), |acc, x| acc * x)
    }
}

impl<F: Field, P: CubicParameters<F>> AddAssign for CubicExtension<F, P> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field, P: CubicParameters<F>> MulAssign for CubicExtension<F, P> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field, P: CubicParameters<F>> MulAssign<F> for CubicExtension<F, P> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: Field, P: CubicParameters<F>> SubAssign for CubicExtension<F, P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Field, P: CubicParameters<F>> Div for CubicExtension<F, P> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: Field, P: CubicParameters<F>> DivAssign for CubicExtension<F, P> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Field + Sample, P: CubicParameters<F>> Sample for CubicExtension<F, P> {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from([F::sample(rng), F::sample(rng), F::sample(rng)])
    }
}

impl<F: Field, P: CubicParameters<F>> Default for CubicExtension<F, P> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Field, P: CubicParameters<F>> Hash for CubicExtension<F, P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_array().hash(state);
    }
}

impl<F: Field, P: CubicParameters<F>> Ring for CubicExtension<F, P> {
    const ONE: Self = Self::ONE;
    const ZERO: Self = Self::ZERO;
}

impl<F: Field, P: CubicParameters<F>> Algebra<F> for CubicExtension<F, P> {}

impl<F: Field, P: CubicParameters<F>> Extension<F> for CubicExtension<F, P> {
    const D: usize = 3;

    fn as_base_slice(&self) -> &[F] {
        self.0.as_slice()
    }

    fn from_base_slice(elements: &[F]) -> Self {
        let mut array = [F::ZERO; 3];
        array.copy_from_slice(elements);
        Self::from(array)
    }
}

impl<F: Field, P: CubicParameters<F>> ExtensionField<F> for CubicExtension<F, P> {}

impl<F: Field, P: CubicParameters<F>> Field for CubicExtension<F, P> {
    fn try_inverse(&self) -> Option<Self> {
        self.try_inverse()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubicElement<T>(pub [T; 3]);

impl<T: Copy> CubicElement<T> {
    #[inline]
    pub const fn new(a: T, b: T, c: T) -> Self {
        Self([a, b, c])
    }

    #[inline]
    pub const fn from_base(element: T, zero: T) -> Self {
        Self([element, zero, zero])
    }

    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        assert_eq!(slice.len(), 3, "Cubic array slice must have length 3");
        Self([slice[0], slice[1], slice[2]])
    }

    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        &self.0
    }

    #[inline]
    pub const fn as_array(&self) -> [T; 3] {
        self.0
    }
}

impl<T: Copy + Add<Output = T>> Add for CubicElement<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl<T: Copy + Sub<Output = T>> Sub for CubicElement<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl<T: Copy + Neg<Output = T>> Neg for CubicElement<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl<T: Copy + AddAssign> AddAssign for CubicElement<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
    }
}

impl<T: Copy + SubAssign> SubAssign for CubicElement<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T>> Mul for CubicElement<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (x_0, x_1, x_2) = (self.0[0], self.0[1], self.0[2]);
        let (y_0, y_1, y_2) = (rhs.0[0], rhs.0[1], rhs.0[2]);

        // Using u^3 = u-1 we get:
        // (x_0 + x_1 u + x_2 u^2) * (y_0 + y_1 u + y_2 u^2)
        // = (x_0y_0 - x_1y_2 - x_2y_1)
        // + (x_0y_1 + x_1y_0 + x_1y_2 + x_2y_1) u
        // + (x_0y_2 + x_1y_1 + x_2y_0) u^2
        Self([
            x_0 * y_0 - x_1 * y_2 - x_2 * y_1,
            x_0 * y_1 + x_1 * y_0 + x_1 * y_2 + x_2 * y_1 - x_2 * y_2,
            x_0 * y_2 + x_1 * y_1 + x_2 * y_0 + x_2 * y_2,
        ])
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T>> Mul<T> for CubicElement<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let (x_0, x_1, x_2) = (self.0[0], self.0[1], self.0[2]);
        Self([x_0 * rhs, x_1 * rhs, x_2 * rhs])
    }
}
