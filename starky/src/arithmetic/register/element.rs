use anyhow::Result;

use super::register::{Register, RegisterSerializable, RegisterSized};
use super::CellType;
use crate::arithmetic::register::memory::MemorySlice;

/// A register for a single element/column in the trace. The value is not constrainted.
#[derive(Debug, Clone, Copy)]
pub struct ElementRegister(MemorySlice);

impl RegisterSerializable for ElementRegister {
    const CELL: Option<CellType> = None;

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

impl Register for ElementRegister {}
