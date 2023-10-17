use serde::{Deserialize, Serialize};

use crate::chip::register::cell::CellType;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::math::field::{Field, PrimeField64};

/// A register that stores a timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeRegister(ElementRegister);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeStamp<T>(T);

impl RegisterSerializable for TimeRegister {
    const CELL: CellType = CellType::Element;

    fn register(&self) -> &MemorySlice {
        self.0.register()
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self(ElementRegister::from_register_unsafe(register))
    }
}

impl RegisterSized for TimeRegister {
    fn size_of() -> usize {
        1
    }
}

impl Register for TimeRegister {
    type Value<T> = TimeStamp<T>;

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        std::slice::from_ref(&value.0)
    }

    fn value_from_slice<T: Clone>(slice: &[T]) -> Self::Value<T> {
        TimeStamp(slice[0].clone())
    }
}

impl TimeStamp<u32> {
    pub fn as_field_timestamp<F: Field>(&self) -> TimeStamp<F> {
        TimeStamp(F::from_canonical_u32(self.0))
    }
}

impl<F: PrimeField64> TimeStamp<F> {
    pub fn as_value(&self) -> TimeStamp<u32> {
        TimeStamp(self.0.as_canonical_u64() as u32)
    }
}
