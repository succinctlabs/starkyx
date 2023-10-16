use serde::{Serialize, Deserialize};

use crate::chip::{register::{element::ElementRegister, memory::MemorySlice}, register::{RegisterSerializable, cell::CellType, RegisterSized, Register}};


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

