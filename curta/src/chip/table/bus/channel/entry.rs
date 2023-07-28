use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::cubic::CubicRegister;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum Entry<F> {
    Input(CubicRegister, ArithmeticExpression<F>),
    Output(CubicRegister, ArithmeticExpression<F>),
}
