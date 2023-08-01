use crate::chip::register::cell::CellType;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};

#[derive(Debug, Clone, Copy)]
pub struct ByteRegister(MemorySlice);

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
