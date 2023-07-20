use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::math::prelude::*;

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

impl<R: Ring + Copy> Product for CubicElement<R> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(CubicElement([R::ONE, R::ZERO, R::ZERO]), |acc, x| acc * x)
    }
}

impl<R: Ring + Copy> Sum for CubicElement<R> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(CubicElement([R::ZERO, R::ZERO, R::ZERO]), |acc, x| acc + x)
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T>> MulAssign for CubicElement<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<R: Ring> Default for CubicElement<R> {
    fn default() -> Self {
        Self([R::ZERO, R::ZERO, R::ZERO])
    }
}

impl<R: Ring + Copy> Ring for CubicElement<R> {
    const ONE: Self = Self([R::ONE, R::ZERO, R::ZERO]);
    const ZERO: Self = Self([R::ZERO, R::ZERO, R::ZERO]);
}
