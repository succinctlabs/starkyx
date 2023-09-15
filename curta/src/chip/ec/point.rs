use num::BigUint;
use serde::{Deserialize, Serialize};

use super::EllipticCurveParameters;
use crate::chip::field::register::FieldRegister;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinePoint<E: EllipticCurveParameters> {
    pub x: BigUint,
    pub y: BigUint,
    _marker: std::marker::PhantomData<E>,
}

impl<E: EllipticCurveParameters> AffinePoint<E> {
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
#[allow(non_snake_case)]
pub struct AffinePointRegister<E: EllipticCurveParameters> {
    pub x: FieldRegister<E::BaseField>,
    pub y: FieldRegister<E::BaseField>,
}

impl<E: EllipticCurveParameters> AffinePointRegister<E> {
    pub fn new(x: FieldRegister<E::BaseField>, y: FieldRegister<E::BaseField>) -> Self {
        Self { x, y }
    }
}
