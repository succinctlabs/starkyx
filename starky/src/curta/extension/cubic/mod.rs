//! An implementation of the cubic extension F[X]/(X^3 - X - 1).
//!
//!

pub mod array;
pub mod cubic_expression;
pub mod expression_constraints;
pub mod gadget;
pub mod goldilocks_cubic;
pub mod register;

use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use plonky2::field::types::{Field, Sample};
use rand::Rng;

/// Parameters for the cubic extension F[X]/(X^3 - X - 1)
pub trait CubicParameters<F>:
    'static + Sized + Copy + Clone + Send + Sync + PartialEq + Eq + std::fmt::Debug
{
    /// The Galois orbit of the generator.
    ///
    /// These are the roots of X^3 - X - 1 in the extension field not equal to X.
    const GALOIS_ORBIT: [[F; 3]; 2];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubicExtension<F: Field, P: CubicParameters<F>>(pub [F; 3], PhantomData<P>);

impl<F: Field, P: CubicParameters<F>> CubicExtension<F, P> {
    const ORBIT: [Self; 2] = [
        Self(P::GALOIS_ORBIT[0], PhantomData),
        Self(P::GALOIS_ORBIT[1], PhantomData),
    ];

    pub const ZERO: Self = Self([F::ZERO, F::ZERO, F::ZERO], PhantomData);
    pub const ONE: Self = Self([F::ONE, F::ZERO, F::ZERO], PhantomData);

    fn in_base_field(&self) -> bool {
        self.0[1] == F::ZERO && self.0[2] == F::ZERO
    }

    pub fn try_inverse(&self) -> Option<Self> {
        let (a, b, c) = (self.0[0], self.0[1], self.0[2]);
        let gal =
            |i: usize| Self::from(a) + Self::ORBIT[i] * b + (Self::ORBIT[i] * Self::ORBIT[i]) * c;
        let (gal_1, gal_2) = (gal(0), gal(1));

        let gal_12 = gal_1 * gal_2;
        let gal_prod = *self * gal_12;
        assert!(gal_prod.in_base_field());

        let gal_inv = gal_prod.0[0].try_inverse()?;
        Some(gal_12 * gal_inv)
    }

    pub fn inverse(&self) -> Self {
        self.try_inverse().expect("Cannot invert zero")
    }
}

impl<F: Field, P: CubicParameters<F>> From<[F; 3]> for CubicExtension<F, P> {
    fn from(value: [F; 3]) -> Self {
        Self([value[0], value[1], value[2]], PhantomData)
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
        Self::from([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl<F: Field, P: CubicParameters<F>> Mul for CubicExtension<F, P> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (x_0, x_1, x_2) = (self.0[0], self.0[1], self.0[2]);
        let (y_0, y_1, y_2) = (rhs.0[0], rhs.0[1], rhs.0[2]);

        // Using u^3 = u-1 we get:
        // (x_0 + x_1 u + x_2 u^2) * (y_0 + y_1 u + y_2 u^2)
        // = (x_0y_0 - x_1y_2 - x_2y_1)
        // + (x_0y_1 + x_1y_0 + x_1y_2 + x_2y_1) u
        // + (x_0y_2 + x_1y_1 + x_2y_0) u^2
        Self::from([
            x_0 * y_0 - x_1 * y_2 - x_2 * y_1,
            x_0 * y_1 + x_1 * y_0 + x_1 * y_2 + x_2 * y_1 - x_2 * y_2,
            x_0 * y_2 + x_1 * y_1 + x_2 * y_0 + x_2 * y_2,
        ])
    }
}

impl<F: Field, P: CubicParameters<F>> Mul<F> for CubicExtension<F, P> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        let (x_0, x_1, x_2) = (self.0[0], self.0[1], self.0[2]);
        Self::from([x_0 * rhs, x_1 * rhs, x_2 * rhs])
    }
}

impl<F: Field, P: CubicParameters<F>> Sub for CubicExtension<F, P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl<F: Field, P: CubicParameters<F>> Neg for CubicExtension<F, P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl<'a, F: Field, P: CubicParameters<F>> Sum<&'a Self> for CubicExtension<F, P> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::from([F::ZERO, F::ZERO, F::ZERO]), |acc, x| acc + *x)
    }
}

impl<'a, F: Field, P: CubicParameters<F>> Product<&'a Self> for CubicExtension<F, P> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::from([F::ONE, F::ZERO, F::ZERO]), |acc, x| acc * *x)
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

impl<F: Field, P: CubicParameters<F>> Sample for CubicExtension<F, P> {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from([F::sample(rng), F::sample(rng), F::sample(rng)])
    }
}
