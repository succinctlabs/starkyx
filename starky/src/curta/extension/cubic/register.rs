use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::cubic_expression::CubicExpression;
use crate::curta::register::{
    ArrayRegister, CellType, ElementRegister, MemorySlice, Register, RegisterSerializable,
    RegisterSized,
};

#[derive(Debug, Clone, Copy)]
pub struct CubicElementRegister {
    coefficients: ArrayRegister<ElementRegister>,
}

impl CubicElementRegister {
    pub fn from_element_array(coefficients: ArrayRegister<ElementRegister>) -> Self {
        assert!(coefficients.len() == 3);
        Self { coefficients }
    }

    pub fn extension_expr<F: RichField + Extendable<D>, const D: usize>(
        &self,
    ) -> CubicExpression<F, D> {
        CubicExpression::from_element_array(self.coefficients)
    }
}

impl RegisterSerializable for CubicElementRegister {
    const CELL: Option<CellType> = ElementRegister::CELL;

    fn register(&self) -> &MemorySlice {
        self.coefficients.register()
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self {
            coefficients: ArrayRegister::from_register_unsafe(register),
        }
    }
}

impl RegisterSized for CubicElementRegister {
    fn size_of() -> usize {
        3
    }
}

impl Register for CubicElementRegister {}
