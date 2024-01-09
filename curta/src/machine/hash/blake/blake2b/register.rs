use serde::{Deserialize, Serialize};

use crate::chip::register::array::{ArrayIterator, ArrayRegister};
use crate::chip::register::cell::CellType;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::chip::uint::register::U64Register;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BLAKE2BDigestRegister(ArrayRegister<U64Register>);

impl RegisterSerializable for BLAKE2BDigestRegister {
    const CELL: CellType = CellType::Element;
    fn register(&self) -> &MemorySlice {
        self.0.register()
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self(ArrayRegister::from_register_unsafe(register))
    }
}

impl RegisterSized for BLAKE2BDigestRegister {
    fn size_of() -> usize {
        U64Register::size_of() * 4
    }
}

impl Register for BLAKE2BDigestRegister {
    type Value<T> = [T; 32];

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        value
    }

    fn value_from_slice<T: Copy>(slice: &[T]) -> Self::Value<T> {
        let elem_fn = |i| slice[i];
        core::array::from_fn(elem_fn)
    }
}

impl BLAKE2BDigestRegister {
    pub fn as_array(&self) -> ArrayRegister<U64Register> {
        self.0
    }
    pub fn split(&self) -> [U64Register; 4] {
        core::array::from_fn(|i| U64Register::from_limbs(&self.0.get_subarray(2 * i..2 * i + 2)))
    }

    pub fn get(&self, index: usize) -> U64Register {
        self.0.get(index)
    }

    pub fn iter(&self) -> ArrayIterator<U64Register> {
        self.0.iter()
    }

    pub fn from_array(array: ArrayRegister<U64Register>) -> Self {
        assert_eq!(array.len(), 4);
        Self(array)
    }
}

impl From<BLAKE2BDigestRegister> for ArrayRegister<U64Register> {
    fn from(value: BLAKE2BDigestRegister) -> Self {
        value.0
    }
}
