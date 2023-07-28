use super::cell::CellType;
use super::memory::MemorySlice;
use super::{Register, RegisterSerializable, RegisterSized};

/// A register for a single element/column in the trace that is supposed to represent a bit. The
/// value is automatically constrained to be 0 or 1 via the quadratic constraint x * (x - 1) == 0.
#[derive(Debug, Clone, Copy)]
pub struct BitRegister(MemorySlice);

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
