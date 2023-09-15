use serde::{Deserialize, Serialize};

use super::array::ArrayRegister;
use super::cell::CellType;
use super::element::ElementRegister;
use super::memory::MemorySlice;
use super::{Register, RegisterSerializable, RegisterSized};
use crate::air::parser::AirParser;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::math::extension::cubic::element::CubicElement;
use crate::math::prelude::*;

/// A register for a single element/column in the trace. The value is not constrainted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CubicRegister(MemorySlice);

pub trait EvalCubic: Register {
    fn value_as_cubic<T: Copy>(value: Self::Value<T>, zero: T) -> CubicElement<T>;

    fn eval_cubic<AP: AirParser>(&self, parser: &mut AP) -> CubicElement<AP::Var> {
        let value = self.eval(parser);
        let zero = parser.zero();
        Self::value_as_cubic(value, zero)
    }

    fn trace_value_as_cubic<F: Field>(value: Self::Value<F>) -> CubicElement<F> {
        let zero = F::ZERO;
        Self::value_as_cubic(value, zero)
    }
}

impl RegisterSerializable for CubicRegister {
    const CELL: CellType = CellType::Element;

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        CubicRegister(register)
    }
}

impl RegisterSized for CubicRegister {
    fn size_of() -> usize {
        3
    }
}

impl Register for CubicRegister {
    type Value<T> = CubicElement<T>;

    fn value_from_slice<T: Copy>(slice: &[T]) -> Self::Value<T> {
        debug_assert!(
            slice.len() == 3,
            "Slice length mismatch for cubic register (expected 3, got {})",
            slice.len()
        );
        CubicElement(core::array::from_fn(|i| slice[i]))
    }

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        &value.0
    }

    fn expr<F: Field>(&self) -> crate::chip::arithmetic::expression::ArithmeticExpression<F> {
        unimplemented!(
            "Cannot create expression from cubic register, use the method ext_expr() instead"
        )
    }
}

impl CubicRegister {
    pub fn as_base_array(&self) -> [ElementRegister; 3] {
        let array = ArrayRegister::<ElementRegister>::from_register_unsafe(*self.register());
        [array.get(0), array.get(1), array.get(2)]
    }

    pub fn ext_expr<F: Field>(&self) -> CubicElement<ArithmeticExpression<F>> {
        let array = ArrayRegister::<ElementRegister>::from_register_unsafe(*self.register());
        CubicElement::new(
            array.get(0).expr(),
            array.get(1).expr(),
            array.get(2).expr(),
        )
    }
}

impl EvalCubic for CubicRegister {
    fn value_as_cubic<T: Copy>(value: Self::Value<T>, _zero: T) -> CubicElement<T> {
        value
    }
}

impl EvalCubic for ElementRegister {
    fn value_as_cubic<T: Copy>(value: Self::Value<T>, zero: T) -> CubicElement<T> {
        CubicElement::from_base(value, zero)
    }
}
