mod array;
mod bit;
mod cell;
mod element;
mod field;
mod memory;
mod u16;

pub use array::RegisterArray;
pub use bit::BitRegister;
pub use cell::CellType;
pub use element::ElementRegister;
pub use field::FieldRegister;
pub use memory::MemorySlice;

pub use self::u16::U16Register;

/// Adds serialization and deserialization to a register for converting between the canonical type
/// and `MemorySlice`.
pub trait RegisterSerializable
where
    Self: Sized,
{
    const CELL: Option<CellType>;

    fn register(&self) -> &MemorySlice;

    /// Initializes the register given a memory slice without checks on length. Use
    /// `from_register` if you want to check the length.
    fn from_register_unsafe(register: MemorySlice) -> Self;

    /// Returns the register but in the next row.
    fn next(&self) -> Self {
        Self::from_register_unsafe(self.register().next())
    }
}

/// Ensures that the register has a fixed size.
pub trait RegisterSized {
    /// Returns the expected size of the register in cells.
    fn size_of() -> usize;
}

/// A register is a slice of memory in the trace that is supposed to represent a specific type of
/// data. A register can be thought as a compiler provided type--it should not be necessary to
/// ever use the register type directly to access values. If you want to access the values, you
/// should instead compose multiple different register types into a struct.
pub trait Register:
    RegisterSerializable + RegisterSized + 'static + Sized + Clone + Send + Sync
{
    /// Initializes the register given a memory slice with checks on length.
    fn from_register(register: MemorySlice) -> Self {
        if register.len() != Self::size_of() {
            panic!("Invalid register length");
        }
        Self::from_register_unsafe(register)
    }
}
