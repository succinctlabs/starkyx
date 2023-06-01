use super::{CellType, Register, RegisterSerializable, RegisterSized};
use crate::curta::air::parser::AirParser;
use crate::curta::register::memory::MemorySlice;

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

impl Register for ElementRegister {
    type Value<AP: AirParser> = AP::Var;

    fn eval<AP: AirParser>(&self, parser: &AP) -> Self::Value<AP> {
        self.register().eval_slice(parser)[0]
    }
}
