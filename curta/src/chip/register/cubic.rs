use super::array::ArrayRegister;
use super::cell::CellType;
use super::element::ElementRegister;
use super::memory::MemorySlice;
use super::{Register, RegisterSerializable, RegisterSized};
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::math::prelude::*;
use crate::plonky2::field::cubic::element::CubicElement;

/// A register for a single element/column in the trace. The value is not constrainted.
#[derive(Debug, Clone, Copy)]
pub struct CubicRegister(MemorySlice);

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

    fn expr<F: Field>(
        &self,
    ) -> crate::chip::constraint::arithmetic::expression::ArithmeticExpression<F> {
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
