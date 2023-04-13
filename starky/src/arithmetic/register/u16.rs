use super::cell::CellType;
use crate::arithmetic::register::memory::MemorySlice;
use crate::arithmetic::register::register::Register;

/// A register for a single element/column in the trace that is supposed to represent a u16. The
/// value is automatically range checked via the lookup table if the register is allocated through
/// the builder.
#[derive(Debug, Clone, Copy)]
pub struct U16Register(MemorySlice);

impl Register for U16Register {
    const CELL: Option<CellType> = Some(CellType::U16);

    fn from_raw_register(register: MemorySlice) -> Self {
        Self(register)
    }

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn size_of() -> usize {
        panic!("Cannot get size of U16Array")
    }

    fn into_raw_register(self) -> MemorySlice {
        self.0
    }
}
