use super::cell::CellType;
use super::{RegisterSerializable, RegisterSized};
use crate::curta::air::parser::AirParser;
use crate::curta::register::memory::MemorySlice;
use crate::curta::register::Register;

/// A register for a single element/column in the trace that is supposed to represent a u16. The
/// value is automatically range checked via the lookup table if the register is allocated through
/// the builder.
#[derive(Debug, Clone, Copy)]
pub struct U16Register(MemorySlice);

impl RegisterSerializable for U16Register {
    const CELL: Option<CellType> = Some(CellType::U16);

    fn register(&self) -> &MemorySlice {
        &self.0
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self(register)
    }
}

impl RegisterSized for U16Register {
    fn size_of() -> usize {
        1
    }
}

impl Register for U16Register {
    type Value<AP: AirParser> = AP::Var;

    fn eval<AP: AirParser>(&self, parser: &AP) -> Self::Value<AP> {
        self.register().eval_slice(parser)[0]
    }
}
