use alloc::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::air::parser::AirParser;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::AirWriter;
use crate::math::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ArithmeticExpressionSlice<F> {
    /// A contiguous chunk of elemnt of a trace column.
    Input(MemorySlice),
    /// A constant vector of field values.
    Const(Vec<F>),
    /// The addition of two arithmetic expressions.
    Add(
        Arc<ArithmeticExpressionSlice<F>>,
        Arc<ArithmeticExpressionSlice<F>>,
    ),
    /// The subtraction of two arithmetic expressions
    Sub(
        Arc<ArithmeticExpressionSlice<F>>,
        Arc<ArithmeticExpressionSlice<F>>,
    ),
    /// The scalar multiplication of an arithmetic expression by a field element.
    ConstMul(F, Arc<ArithmeticExpressionSlice<F>>),
    /// The scalar multiplication of an arithmetic expression by an arithmetic expression of size 1
    ScalarMul(
        Arc<ArithmeticExpressionSlice<F>>,
        Arc<ArithmeticExpressionSlice<F>>,
    ),
    /// The multiplication of two arithmetic expressions.
    Mul(
        Arc<ArithmeticExpressionSlice<F>>,
        Arc<ArithmeticExpressionSlice<F>>,
    ),
}

impl<F: Field> ArithmeticExpressionSlice<F> {
    pub fn from_raw_register(input: MemorySlice) -> Self {
        ArithmeticExpressionSlice::Input(input)
    }

    pub fn registers(&self) -> Vec<MemorySlice> {
        match self {
            ArithmeticExpressionSlice::Input(input) => vec![*input],
            ArithmeticExpressionSlice::Const(_) => vec![],
            ArithmeticExpressionSlice::Add(left, right) => {
                let mut left = left.registers();
                let mut right = right.registers();
                left.append(&mut right);
                left
            }
            ArithmeticExpressionSlice::Sub(left, right) => {
                let mut left = left.registers();
                let mut right = right.registers();
                left.append(&mut right);
                left
            }
            ArithmeticExpressionSlice::ConstMul(_, expr) => expr.registers(),
            ArithmeticExpressionSlice::ScalarMul(left, right) => {
                let mut left = left.registers();
                let mut right = right.registers();
                left.append(&mut right);
                left
            }
            ArithmeticExpressionSlice::Mul(left, right) => {
                let mut left = left.registers();
                let mut right = right.registers();
                left.append(&mut right);
                left
            }
        }
    }

    pub(crate) fn read_from_slice(&self, slice: &[F]) -> Vec<F> {
        match self {
            ArithmeticExpressionSlice::Input(input) => input.read_from_slice(slice).to_vec(),
            ArithmeticExpressionSlice::Const(constants) => constants.clone(),
            ArithmeticExpressionSlice::Add(left, right) => {
                let left = left.read_from_slice(slice);
                let right = right.read_from_slice(slice);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *l + *r)
                    .collect()
            }
            ArithmeticExpressionSlice::Sub(left, right) => {
                let left = left.read_from_slice(slice);
                let right = right.read_from_slice(slice);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *l - *r)
                    .collect()
            }
            ArithmeticExpressionSlice::ConstMul(scalar, expr) => {
                let expr_val = expr.read_from_slice(slice);
                expr_val.iter().map(|x| *x * *scalar).collect()
            }
            ArithmeticExpressionSlice::ScalarMul(scalar, expr) => {
                let scalar_val = scalar.read_from_slice(slice)[0];
                let expr_val = expr.read_from_slice(slice);
                expr_val.iter().map(|x| *x * scalar_val).collect()
            }
            ArithmeticExpressionSlice::Mul(left, right) => {
                let left_vals = left.read_from_slice(slice);
                let right_vals = right.read_from_slice(slice);
                left_vals
                    .iter()
                    .zip(right_vals.iter())
                    .map(|(l, r)| *l * *r)
                    .collect()
            }
        }
    }

    pub fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) -> Vec<AP::Var> {
        match self {
            ArithmeticExpressionSlice::Input(input) => input.eval_slice(parser).to_vec(),
            ArithmeticExpressionSlice::Const(constants) => {
                constants.iter().map(|x| parser.constant(*x)).collect()
            }
            ArithmeticExpressionSlice::Add(left, right) => {
                let left = left.eval(parser);
                let right = right.eval(parser);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| parser.add(*l, *r))
                    .collect()
            }
            ArithmeticExpressionSlice::Sub(left, right) => {
                let left = left.eval(parser);
                let right = right.eval(parser);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| parser.sub(*l, *r))
                    .collect()
            }
            ArithmeticExpressionSlice::ConstMul(scalar, expr) => {
                let expr_val = expr.eval(parser);
                expr_val
                    .iter()
                    .map(|x| parser.mul_const(*x, *scalar))
                    .collect()
            }
            ArithmeticExpressionSlice::ScalarMul(scalar, expr) => {
                let scalar_val = scalar.eval(parser)[0];
                let expr_val = expr.eval(parser);
                expr_val
                    .iter()
                    .map(|x| parser.mul(*x, scalar_val))
                    .collect()
            }
            ArithmeticExpressionSlice::Mul(left, right) => {
                let left_vals = left.eval(parser);
                let right_vals = right.eval(parser);
                left_vals
                    .iter()
                    .zip(right_vals.iter())
                    .map(|(l, r)| parser.mul(*l, *r))
                    .collect()
            }
        }
    }

    pub fn eval_writer(&self, writer: &impl AirWriter<Field = F>) -> Vec<F> {
        match self {
            ArithmeticExpressionSlice::Input(input) => writer.read_slice(input).to_vec(),
            ArithmeticExpressionSlice::Const(constants) => constants.to_vec(),
            ArithmeticExpressionSlice::Add(left, right) => {
                let left = left.eval_writer(writer);
                let right = right.eval_writer(writer);
                left.into_iter().zip(right).map(|(l, r)| l + r).collect()
            }
            ArithmeticExpressionSlice::Sub(left, right) => {
                let left = left.eval_writer(writer);
                let right = right.eval_writer(writer);
                left.into_iter().zip(right).map(|(l, r)| l - r).collect()
            }
            ArithmeticExpressionSlice::ConstMul(scalar, expr) => {
                let expr_val = expr.eval_writer(writer);
                expr_val.into_iter().map(|x| x * *scalar).collect()
            }
            ArithmeticExpressionSlice::ScalarMul(scalar, expr) => {
                let scalar_val = scalar.eval_writer(writer)[0];
                let expr_val = expr.eval_writer(writer);
                expr_val.into_iter().map(|x| x * scalar_val).collect()
            }
            ArithmeticExpressionSlice::Mul(left, right) => {
                let left_vals = left.eval_writer(writer);
                let right_vals = right.eval_writer(writer);
                left_vals
                    .into_iter()
                    .zip(right_vals)
                    .map(|(l, r)| l * r)
                    .collect()
            }
        }
    }
}
