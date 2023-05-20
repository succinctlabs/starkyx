use core::ops::{Add, Mul, Sub};

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::{CubicExtension, CubicParameters};
use crate::curta::constraint::arithmetic::ArithmeticExpression;
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

    pub fn from_constants(array: [F; 3]) -> Self {
        Self([
            ArithmeticExpression::from_constant(array[0]),
            ArithmeticExpression::from_constant(array[1]),
            ArithmeticExpression::from_constant(array[2]),
        ])
    }

    pub fn into_expressions_array(self) -> [ArithmeticExpression<F, D>; 3] {
        self.0
    }

    pub fn eval<P: CubicParameters<F>>(
        &self,
        trace_rows: &[Vec<F>],
        row_index: usize,
    ) -> CubicExtension<F, P> {
        let value = (
            self.0[0].eval(trace_rows, row_index),
            self.0[1].eval(trace_rows, row_index),
            self.0[2].eval(trace_rows, row_index),
        );
        assert_eq!(value.0.len(), 1);
        assert_eq!(value.1.len(), 1);
        assert_eq!(value.2.len(), 1);

        CubicExtension::from([value.0[0], value.1[0], value.2[0]])
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

impl<F: RichField + Extendable<D>, const D: usize> From<ArithmeticExpression<F, D>>
    for CubicExpression<F, D>
{
    fn from(expression: ArithmeticExpression<F, D>) -> Self {
        Self([
            expression,
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

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField as F;
    use plonky2::field::types::Sample;

    use super::super::goldilocks_cubic::GF3;
    use super::*;
    use crate::curta::extension::cubic::goldilocks_cubic::GoldilocksCubicParameters;
    use crate::curta::register::{MemorySlice, RegisterSerializable};

    #[test]
    fn test_cubic_expression() {
        type P = GoldilocksCubicParameters;
        let num_tests = 100;

        let a_reg =
            ArrayRegister::<ElementRegister>::from_register_unsafe(MemorySlice::Local(0, 3));
        let b_reg =
            ArrayRegister::<ElementRegister>::from_register_unsafe(MemorySlice::Local(3, 3));
        let c_reg =
            ArrayRegister::<ElementRegister>::from_register_unsafe(MemorySlice::Local(6, 3));

        let a = CubicExpression::<F, 1>::from_element_array(a_reg);
        let b = CubicExpression::<F, 1>::from_element_array(b_reg);
        let c = CubicExpression::<F, 1>::from_element_array(c_reg);

        let expr_a_p_b = a.clone() + b.clone();
        let expr_a_m_b = a.clone() - b.clone();
        let expr_ab = a.clone() * b.clone();
        let expr_ab_p_c = a.clone() * b.clone() + c.clone();

        for _ in 0..num_tests {
            let a_v = GF3::rand();
            let b_v = GF3::rand();
            let c_v = GF3::rand();

            let row = a_v
                .base_field_array()
                .iter()
                .chain(b_v.base_field_array().iter())
                .chain(c_v.base_field_array().iter())
                .map(|x| *x)
                .collect::<Vec<_>>();
            let trace = vec![row];

            let a_eval = a.eval::<P>(&trace, 0);
            let b_eval = b.eval::<P>(&trace, 0);
            let c_eval = c.eval::<P>(&trace, 0);

            assert_eq!(a_eval, a_v);
            assert_eq!(b_eval, b_v);
            assert_eq!(c_eval, c_v);

            let expr_a_p_b_eval = expr_a_p_b.eval::<P>(&trace, 0);
            let expr_a_m_b_eval = expr_a_m_b.eval::<P>(&trace, 0);
            let expr_ab_eval = expr_ab.eval::<P>(&trace, 0);
            let expr_ab_p_c_eval = expr_ab_p_c.eval::<P>(&trace, 0);

            assert_eq!(expr_a_p_b_eval, a_v + b_v);
            assert_eq!(expr_a_m_b_eval, a_v - b_v);
            assert_eq!(expr_ab_eval, a_v * b_v);
            assert_eq!(expr_ab_p_c_eval, a_v * b_v + c_v);
        }
    }
}
