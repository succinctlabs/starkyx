use crate::arithmetic::register2::memory::MemorySlice;
use crate::arithmetic::register2::register::{Register, RegisterType};

/// A register for a single element/column in the trace. The value is not constrainted.
#[derive(Debug, Clone, Copy)]
pub struct ElementRegister(MemorySlice);

impl Register for ElementRegister {
    const CELL: Option<RegisterType> = None;

    fn from_raw_register(register: MemorySlice) -> Self {
        ElementRegister(register)
    }

    fn into_raw_register(self) -> MemorySlice {
        self.0
    }

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn size_of() -> usize {
        1
    }
}
