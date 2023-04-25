use num::BigUint;

use crate::arithmetic::parameters::EllipticCurveParameters;

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
