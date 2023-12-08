use serde::{Deserialize, Serialize};

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::math::prelude::*;

/// A register that stores a timestamp.
pub type Time<F> = TimeStamp<ArithmeticExpression<F>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeStamp<T>(pub(crate) T);

impl<T> TimeStamp<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(value)
    }
}

impl<F: Field> Time<F> {
    pub fn zero() -> Self {
        Self::new(ArithmeticExpression::zero())
    }

    pub fn constant(value: usize) -> Self {
        Self::new(ArithmeticExpression::from(F::from_canonical_usize(value)))
    }

    pub fn from_element(element: ElementRegister) -> Self {
        Self::new(element.expr())
    }

    pub fn expr(&self) -> ArithmeticExpression<F> {
        self.0.clone()
    }

    pub fn advance_by(&self, interval: usize) -> Self {
        Self::new(self.0.clone() + ArithmeticExpression::from(F::from_canonical_usize(interval)))
    }

    pub fn advance(&self) -> Self {
        self.advance_by(1)
    }

    pub fn decrement_by(&self, interval: usize) -> Self {
        Self::new(self.0.clone() - ArithmeticExpression::from(F::from_canonical_usize(interval)))
    }

    pub fn decrement(&self) -> Self {
        self.decrement_by(1)
    }
}
