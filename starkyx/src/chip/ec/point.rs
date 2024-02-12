use core::ops::{Add, Neg};

use num::BigUint;
use serde::{Deserialize, Serialize};

use super::EllipticCurve;
use crate::chip::field::register::FieldRegister;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinePoint<E> {
    pub x: BigUint,
    pub y: BigUint,
    _marker: std::marker::PhantomData<E>,
}

impl<E> AffinePoint<E> {
    #[allow(dead_code)]
    pub fn new(x: BigUint, y: BigUint) -> Self {
        Self {
            x,
            y,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AffinePointRegister<E: EllipticCurve> {
    pub x: FieldRegister<E::BaseField>,
    pub y: FieldRegister<E::BaseField>,
}

impl<E: EllipticCurve> AffinePointRegister<E> {
    pub fn new(x: FieldRegister<E::BaseField>, y: FieldRegister<E::BaseField>) -> Self {
        Self { x, y }
    }
}

impl<E: EllipticCurve> Add<&AffinePoint<E>> for &AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn add(self, other: &AffinePoint<E>) -> AffinePoint<E> {
        E::ec_add(self, other)
    }
}

impl<E: EllipticCurve> Add<AffinePoint<E>> for AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn add(self, other: AffinePoint<E>) -> AffinePoint<E> {
        &self + &other
    }
}

impl<E: EllipticCurve> Add<&AffinePoint<E>> for AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn add(self, other: &AffinePoint<E>) -> AffinePoint<E> {
        &self + other
    }
}

impl<E: EllipticCurve> Neg for &AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn neg(self) -> AffinePoint<E> {
        E::ec_neg(self)
    }
}

impl<E: EllipticCurve> Neg for AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn neg(self) -> AffinePoint<E> {
        -&self
    }
}
