use alloc::sync::Arc;
use core::ops::{Add, Mul, Sub};

use serde::{Deserialize, Serialize};

use super::expression_slice::ArithmeticExpressionSlice;
use crate::air::parser::AirParser;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;
/// An abstract representation of an arithmetic expression.
///
/// An arithmetic expression is a vector of polynomials in the trace columns, i.e.,
/// [ P_1(q_1(x), q_2(x), ..., q_n(x)), ..., P_n(q_1(x), q_2(x), ..., q_n(x))]
///
/// Operations on Arithmetic expressions are done pointwise. For arithmetic expressions:
/// P = [P_1, ..., P_n] and Q = [Q_1, ..., Q_n], we define:
/// - P + Q = [P_1 + Q_1, ..., P_n + Q_n]
/// - P - Q = [P_1 - Q_1, ..., P_n - Q_n]
/// - c * P = [c * P_1, ..., c * P_n] for c in F
///
/// If Z = [Z_1] is a vector of length 1, we also define
/// - P * Z = [P_1 * Z_1, ..., P_n * Z_1]
///
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArithmeticExpression<F> {
    pub(crate) expression: ArithmeticExpressionSlice<F>,
    pub size: usize,
    // pub degree : usize,
}

impl<F: Field> ArithmeticExpression<F> {
    pub fn from_constant_vec(constants: Vec<F>) -> Self {
        let size = constants.len();
        Self {
            expression: ArithmeticExpressionSlice::Const(constants),
            size,
        }
    }

    pub fn from_constant(constant: F) -> Self {
        Self::from_constant_vec(vec![constant])
    }

    pub fn zero() -> Self {
        Self::from_constant(F::ZERO)
    }

    pub fn one() -> Self {
        Self::from_constant(F::ONE)
    }

    pub fn read_from_slice(&self, slice: &[F]) -> Vec<F> {
        self.expression.read_from_slice(slice)
    }

    pub fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) -> Vec<AP::Var> {
        self.expression.eval(parser)
    }

    /// Returns the registers used in the expression.
    pub fn registers(&self) -> Vec<MemorySlice> {
        self.expression.registers()
    }

    /// Returns true if any of the registers in the expression is a trace register.
    pub fn is_trace(&self) -> bool {
        !self.registers().iter().all(|reg| !reg.is_trace())
    }
}

impl<F: Field> Add for ArithmeticExpression<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "Cannot add arithmetic expressions of different sizes"
        );
        Self {
            expression: ArithmeticExpressionSlice::Add(
                Arc::new(self.expression),
                Arc::new(rhs.expression),
            ),
            size: self.size,
        }
    }
}

impl<F: Field> From<F> for ArithmeticExpression<F> {
    fn from(f: F) -> Self {
        Self::from_constant(f)
    }
}

impl<F: Field> Sub for ArithmeticExpression<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "Cannot subtract arithmetic expressions of different sizes"
        );
        Self {
            expression: ArithmeticExpressionSlice::Sub(
                Arc::new(self.expression),
                Arc::new(rhs.expression),
            ),
            size: self.size,
        }
    }
}

impl<F: Field> Add<Vec<F>> for ArithmeticExpression<F> {
    type Output = Self;

    fn add(self, rhs: Vec<F>) -> Self::Output {
        assert_eq!(
            self.size,
            rhs.len(),
            "Cannot add vector of size {} arithmetic expression of size {}",
            rhs.len(),
            self.size
        );
        Self {
            expression: ArithmeticExpressionSlice::Add(
                Arc::new(self.expression),
                Arc::new(ArithmeticExpressionSlice::Const(rhs)),
            ),
            size: self.size,
        }
    }
}

impl<F: Field> Sub<Vec<F>> for ArithmeticExpression<F> {
    type Output = Self;

    fn sub(self, rhs: Vec<F>) -> Self::Output {
        assert_eq!(
            self.size,
            rhs.len(),
            "Cannot subtract a vector of size {} arithmetic expression of size {}",
            rhs.len(),
            self.size
        );
        Self {
            expression: ArithmeticExpressionSlice::Sub(
                Arc::new(self.expression),
                Arc::new(ArithmeticExpressionSlice::Const(rhs)),
            ),
            size: self.size,
        }
    }
}

impl<F: Field> Add<F> for ArithmeticExpression<F> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        self + vec![rhs]
    }
}

impl<F: Field> Sub<F> for ArithmeticExpression<F> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        self - vec![rhs]
    }
}

impl<F: Field> Mul<F> for ArithmeticExpression<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self {
            expression: ArithmeticExpressionSlice::ConstMul(rhs, Arc::new(self.expression)),
            size: self.size,
        }
    }
}

impl<F: Field> Mul for ArithmeticExpression<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.size, rhs.size) {
            (1, _) => Self {
                expression: ArithmeticExpressionSlice::ScalarMul(
                    Arc::new(self.expression),
                    Arc::new(rhs.expression),
                ),
                size: rhs.size,
            },
            (_, 1) => Self {
                expression: ArithmeticExpressionSlice::ScalarMul(
                    Arc::new(rhs.expression),
                    Arc::new(self.expression),
                ),
                size: self.size,
            },
            (n, m) if n == m => Self {
                expression: ArithmeticExpressionSlice::Mul(
                    Arc::new(self.expression),
                    Arc::new(rhs.expression),
                ),
                size: n,
            },
            _ => panic!("Cannot multiply arithmetic expressions of different sizes"),
        }
    }
}
