use serde::{Deserialize, Serialize};

use super::array::ArrayRegister;
use super::cell::CellType;
use super::cubic::CubicRegister;
use super::element::ElementRegister;
use super::memory::MemorySlice;
use super::{Register, RegisterSerializable, RegisterSized};
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::memory::pointer::raw::RawPointer;
use crate::chip::memory::time::Time;
use crate::chip::memory::value::MemoryValue;
use crate::machine::builder::ops::{Add, And, Mul, Not, Or};
use crate::machine::builder::Builder;
use crate::math::prelude::*;

/// A register for a single element/column in the trace that is supposed to represent a bit. The
/// value is automatically constrained to be 0 or 1 via the quadratic constraint x * (x - 1) == 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BitRegister(MemorySlice);

impl BitRegister {
    pub fn not_expr<F: Field>(&self) -> ArithmeticExpression<F> {
        ArithmeticExpression::one() - self.expr()
    }

    #[inline]
    pub fn as_element(&self) -> ElementRegister {
        ElementRegister::from_register_unsafe(self.0)
    }
}

impl RegisterSerializable for BitRegister {
    const CELL: CellType = CellType::Bit;

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        BitRegister(register)
    }
}

impl RegisterSized for BitRegister {
    fn size_of() -> usize {
        1
    }
}

impl Register for BitRegister {
    type Value<T> = T;

    fn value_from_slice<T: Copy>(slice: &[T]) -> Self::Value<T> {
        slice[0]
    }

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        std::slice::from_ref(value)
    }
}

impl MemoryValue for BitRegister {
    fn num_challenges() -> usize {
        0
    }

    fn compress<L: crate::chip::AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        ptr: RawPointer,
        time: &Time<L::Field>,
        challenges: &ArrayRegister<CubicRegister>,
    ) -> CubicRegister {
        self.as_element().compress(builder, ptr, time, challenges)
    }
}

impl<B: Builder> Add<B> for BitRegister {
    type Output = Self;

    fn add(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.expression(self.expr() + rhs.expr())
    }
}

impl<B: Builder> Mul<B> for BitRegister {
    type Output = Self;

    fn mul(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.expression(self.expr() * rhs.expr())
    }
}

impl<B: Builder> Not<B> for BitRegister {
    type Output = Self;

    fn not(self, builder: &mut B) -> Self::Output {
        builder.expression(self.not_expr())
    }
}

impl<B: Builder> Or<B> for BitRegister {
    type Output = Self;

    fn or(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.expression(self.expr() + rhs.expr() - self.expr() * rhs.expr())
    }
}

impl<B: Builder> And<B> for BitRegister {
    type Output = Self;

    fn and(self, rhs: Self, builder: &mut B) -> Self::Output {
        builder.mul(self, rhs)
    }
}
