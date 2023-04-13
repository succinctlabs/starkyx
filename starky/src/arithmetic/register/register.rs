use anyhow::{anyhow, Result};

use super::cell::CellType;
use crate::arithmetic::register::memory::MemorySlice;

pub trait Register: 'static + Sized + Clone + Send + Sync {
    const CELL: Option<CellType>;

    /// Returns an element of the field
    ///
    /// Assumes register is of the correct size
    fn from_raw_register(register: MemorySlice) -> Self;

    fn into_raw_register(self) -> MemorySlice;

    fn register(&self) -> &MemorySlice;

    /// Returns an element of the field
    ///
    /// Checks that the register is of the correct size
    fn from_register(register: MemorySlice) -> Result<Self> {
        if register.len() != Self::size_of() {
            return Err(anyhow!("Invalid register length"));
        }

        Ok(Self::from_raw_register(register))
    }

    fn next(&self) -> Self {
        Self::from_raw_register(self.register().next())
    }

    fn size_of() -> usize;
}
