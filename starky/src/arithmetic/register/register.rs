use anyhow::{anyhow, Result};

use super::cell::CellType;
use crate::arithmetic::register::memory::MemorySlice;

/// A register is a slice of memory in the trace that is supposed to represent a specific type of
/// data. A register can be thought as a compiler provided type--it should not be necessary to
/// ever use the register type directly to access values. If you want to access the values, you
/// should instead compose multiple different register types into a struct.
pub trait Register: 'static + Sized + Clone + Send + Sync {
    /// The type of each cell in the register. Useful for specific constraints we want to apply
    /// on cells such as them falling in a specific range.
    const CELL: Option<CellType>;

    /// Returns the memory slice of the register.
    fn register(&self) -> &MemorySlice;

    /// Returns the register but in the next row.
    fn next(&self) -> Self {
        Self::from_raw_register(self.register().next())
    }

    /// Returns the expected size of the register in cells.
    fn size_of() -> usize;

    /// Initializes the register given a memory slice with no checks on length. Avoid using this
    /// function unless you know what you are doing and use `from_register` instead.
    fn from_raw_register(register: MemorySlice) -> Self;

    /// Initializes the register given a memory slice with checks on length.
    fn from_register(register: MemorySlice) -> Result<Self> {
        if register.len() != Self::size_of() {
            return Err(anyhow!("Invalid register length"));
        }
        Ok(Self::from_raw_register(register))
    }
}
