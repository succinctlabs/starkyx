mod bit;
mod element;
mod field;
mod memory;
mod register;
mod u16;

pub use bit::BitRegister;
pub use element::ElementRegister;
pub use field::FieldRegister;
pub use memory::MemorySlice;
pub use register::{Register, RegisterType};

pub use self::u16::U16Register;
