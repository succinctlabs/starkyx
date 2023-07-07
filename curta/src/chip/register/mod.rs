use self::cell::CellType;
use self::memory::MemorySlice;
use crate::air::parser::AirParser;

pub mod array;
pub mod bit;
pub mod cell;
pub mod element;
pub mod memory;
pub mod u16;

/// Adds serialization and deserialization to a register for converting between the canonical type
/// and `MemorySlice`.
pub trait RegisterSerializable
where
    Self: Sized,
{
    const CELL: CellType;

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
    RegisterSerializable + RegisterSized + 'static + Sized + Clone + Send + Sync + Copy
{
    type Value<T>;

    fn eval<AP: AirParser>(&self, parser: &AP) -> Self::Value<AP::Var>;

    /// Initializes the register given a memory slice with checks on length.
    fn from_register(register: MemorySlice) -> Self {
        if register.len() != Self::size_of() {
            panic!("Invalid register length");
        }
        Self::from_register_unsafe(register)
    }
    // fn expr<F: RichField + Extendable<D>, const D: usize>(&self) -> ArithmeticExpression<F, D> {
    //     ArithmeticExpression {
    //         expression: ArithmeticExpressionSlice::from_raw_register(*self.register()),
    //         size: Self::size_of(),
    //     }
    // }
}
