use serde::{Deserialize, Serialize};

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::cubic::CubicRegister;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Entry<F> {
    Input(CubicRegister, ArithmeticExpression<F>),
    Output(CubicRegister, ArithmeticExpression<F>),
}
