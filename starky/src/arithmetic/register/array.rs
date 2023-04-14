use core::marker::PhantomData;

use anyhow::Result;

use super::register::{Register, RegisterSerializable};
use super::CellType;
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
    pub fn new(register: MemorySlice) -> Result<Self> {
        let length = register.len() / T::size_of();
        Ok(Self {
            register,
            length,
            _marker: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.length
    }
}
