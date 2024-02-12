use serde::{Deserialize, Serialize};

use crate::chip::register::cell::CellType;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ByteRegister(MemorySlice);

impl ByteRegister {
    pub fn element(&self) -> ElementRegister {
        ElementRegister::from_register_unsafe(self.0)
    }
}

impl RegisterSerializable for ByteRegister {
    const CELL: CellType = CellType::Element;

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self(register)
    }
}

impl RegisterSized for ByteRegister {
    fn size_of() -> usize {
        1
    }
}

impl Register for ByteRegister {
    type Value<T> = T;

    fn value_from_slice<T: Copy>(slice: &[T]) -> Self::Value<T> {
        slice[0]
    }

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        core::slice::from_ref(value)
    }
}
