use core::ops::{Add, Mul, Sub};

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use crate::curta::constraint::expression::ArithmeticExpression;
use crate::curta::register::{ArrayRegister, ElementRegister, Register};

/// An element of the cubic extension
/// F[X] / (X^3 - X + 1)
#[derive(Clone, Debug)]
pub struct CubicExpression<F, const D: usize>(pub [ArithmeticExpression<F, D>; 3]);

impl<F: RichField + Extendable<D>, const D: usize> CubicExpression<F, D> {
    pub fn from_elements(arr: [ElementRegister; 3]) -> Self {
        Self([arr[0].expr(), arr[1].expr(), arr[2].expr()])
    }

    pub fn from_element_array(arr: ArrayRegister<ElementRegister>) -> Self {
        assert_eq!(arr.len(), 3);
        Self([arr.get(0).expr(), arr.get(1).expr(), arr.get(2).expr()])
    }
}

impl<F: RichField + Extendable<D>, const D: usize> From<ElementRegister> for CubicExpression<F, D> {
    fn from(register: ElementRegister) -> Self {
        Self([
            register.expr(),
            ArithmeticExpression::zero(),
            ArithmeticExpression::zero(),
        ])
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Add for CubicExpression<F, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0].clone() + rhs.0[0].clone(),
            self.0[1].clone() + rhs.0[1].clone(),
            self.0[2].clone() + rhs.0[2].clone(),
        ])
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Sub for CubicExpression<F, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0].clone() - rhs.0[0].clone(),
            self.0[1].clone() - rhs.0[1].clone(),
            self.0[2].clone() - rhs.0[2].clone(),
        ])
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Mul for CubicExpression<F, D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (x_0, x_1, x_2) = (self.0[0].clone(), self.0[1].clone(), self.0[2].clone());
        let (y_0, y_1, y_2) = (rhs.0[0].clone(), rhs.0[1].clone(), rhs.0[2].clone());

        // Using u^3 = u-1 we get:
        // (x_0 + x_1 u + x_2 u^2) * (y_0 + y_1 u + y_2 u^2)
        // = (x_0y_0 - x_1y_2 - x_2y_1)
        // + (x_0y_1 + x_1y_0 + x_1y_2 + x_2y_1) u
        // + (x_0y_2 + x_1y_1 + x_2y_0) u^2
        Self([
            x_0.clone() * y_0.clone() - x_1.clone() * y_2.clone() - x_2.clone() * y_1.clone(),
            x_0.clone() * y_1.clone()
                + x_1.clone() * y_0.clone()
                + x_1.clone() * y_2.clone()
                + x_2.clone() * y_1.clone()
                - x_2.clone() * y_2.clone(),
            x_0.clone() * y_2.clone()
                + x_1.clone() * y_1.clone()
                + x_2.clone() * y_0.clone()
                + x_2.clone() * y_2.clone(),
        ])
    }
}
