use core::marker::PhantomData;

use super::{CellType, Register, RegisterSerializable};
use crate::arithmetic::register::memory::MemorySlice;

/// A helper struct for representing an array of registers. In particular, it makes it easier
/// to access the memory slice as well as converting from a memory slice to the struct.
#[derive(Debug, Clone, Copy)]
pub struct Array<T: Register> {
    register: MemorySlice,
    length: usize,
    _marker: PhantomData<T>,
}

impl<T: Register> RegisterSerializable for Array<T> {
    const CELL: Option<CellType> = T::CELL;

    fn register(&self) -> &MemorySlice {
        &self.register
    }

    fn from_register_unsafe(register: MemorySlice) -> Self {
        let length = register.len() / T::size_of();
        Self {
            register,
            length,
            _marker: PhantomData,
        }
    }
}

impl<T: Register> Array<T> {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn get(&self, idx: usize) -> T {
        if idx >= self.len() {
            panic!("Index out of bounds");
        }
        let offset = T::size_of() * idx;
        match self.register {
            MemorySlice::Local(col, _) => {
                T::from_register(MemorySlice::Local(col + offset, T::size_of()))
            }
            MemorySlice::Next(col, _) => {
                T::from_register(MemorySlice::Next(col + offset, T::size_of()))
            }
            MemorySlice::First(col, _) => {
                T::from_register(MemorySlice::First(col + offset, T::size_of()))
            }
            MemorySlice::Last(col, _) => {
                T::from_register(MemorySlice::Last(col + offset, T::size_of()))
            }
        }
    }
}
