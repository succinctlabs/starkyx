use serde::{Deserialize, Serialize};

use crate::chip::register::array::{ArrayIterator, ArrayRegister};
use crate::chip::register::cell::CellType;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable, RegisterSized};
use crate::chip::uint::register::{U32Register, U64Register};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SHA256DigestRegister(ArrayRegister<U32Register>);

impl RegisterSerializable for SHA256DigestRegister {
    const CELL: CellType = CellType::Element;
    fn register(&self) -> &MemorySlice {
        self.0.register()
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        Self(ArrayRegister::from_register_unsafe(register))
    }
}

impl RegisterSized for SHA256DigestRegister {
    fn size_of() -> usize {
        U32Register::size_of() * 8
    }
}

impl Register for SHA256DigestRegister {
    type Value<T> = [T; 32];

    fn align<T>(value: &Self::Value<T>) -> &[T] {
        value
    }

    fn value_from_slice<T: Copy>(slice: &[T]) -> Self::Value<T> {
        let elem_fn = |i| slice[i];
        core::array::from_fn(elem_fn)
    }
}

impl SHA256DigestRegister {
    pub fn as_array(&self) -> ArrayRegister<U32Register> {
        self.0
    }
    pub fn split(&self) -> [U64Register; 4] {
        core::array::from_fn(|i| U64Register::from_limbs(&self.0.get_subarray(2 * i..2 * i + 2)))
    }

    pub fn get(&self, index: usize) -> U32Register {
        self.0.get(index)
    }

    pub fn iter(&self) -> ArrayIterator<U32Register> {
        self.0.iter()
    }

    pub fn from_array(array: ArrayRegister<U32Register>) -> Self {
        assert_eq!(array.len(), 8);
        Self(array)
    }
}

impl From<SHA256DigestRegister> for ArrayRegister<U32Register> {
    fn from(value: SHA256DigestRegister) -> Self {
        value.0
    }
}
