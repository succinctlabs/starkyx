use super::cell::CellType;
use super::memory::MemorySlice;
use super::{Register, RegisterSerializable, RegisterSized};
use crate::air::parser::AirParser;

/// A register for a single element/column in the trace that is supposed to represent a u16. The
/// value is automatically range checked via the lookup table if the register is allocated through
/// the builder.
#[derive(Debug, Clone, Copy)]
pub struct U16Register(MemorySlice);

impl RegisterSerializable for U16Register {
    const CELL: CellType = CellType::U16;

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
    type Value<T> = T;

    fn eval<AP: AirParser>(&self, parser: &AP) -> Self::Value<AP::Var> {
        self.register().eval_slice(parser)[0]
    }

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        std::slice::from_ref(value)
    }
}
