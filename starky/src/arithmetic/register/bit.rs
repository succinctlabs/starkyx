use anyhow::Result;

use super::register::{RegisterSerializable, RegisterSized};
use super::CellType;
use crate::arithmetic::register::memory::MemorySlice;
use crate::arithmetic::register::register::Register;

/// A register for a single element/column in the trace that is supposed to represent a bit. The
/// value is automatically constrained to be 0 or 1 via the quadratic constraint x * (x - 1) == 0.
#[derive(Debug, Clone, Copy)]
pub struct BitRegister(MemorySlice);

impl RegisterSerializable for BitRegister {
    const CELL: Option<CellType> = Some(CellType::Bit);

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

impl Register for BitRegister {}
