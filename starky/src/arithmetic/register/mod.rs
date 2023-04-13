mod array;
mod bit;
mod cell;
mod element;
mod field;
mod memory;
mod register;
mod u16;
mod witness;

pub use array::Array;
pub use bit::BitRegister;
pub use cell::CellType;
pub use element::ElementRegister;
pub use field::FieldRegister;
pub use memory::MemorySlice;
pub use register::Register;
pub use witness::WitnessData;

pub use self::u16::U16Register;
