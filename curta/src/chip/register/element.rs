use super::cell::CellType;
use super::memory::MemorySlice;
use super::{Register, RegisterSerializable, RegisterSized};

/// A register for a single element/column in the trace. The value is not constrainted.
#[derive(Debug, Clone, Copy)]
pub struct ElementRegister(MemorySlice);

impl RegisterSerializable for ElementRegister {
    const CELL: CellType = CellType::Element;

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        ElementRegister(register)
    }
}

impl RegisterSized for ElementRegister {
    fn size_of() -> usize {
        1
    }
}

impl Register for ElementRegister {
    type Value<T> = T;

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        std::slice::from_ref(value)
    }

    fn value_from_slice<T: Clone>(slice: &[T]) -> Self::Value<T> {
        slice[0].clone()
    }
}
