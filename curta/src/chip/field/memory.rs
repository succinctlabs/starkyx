// use serde::{Deserialize, Serialize};

// use crate::chip::builder::AirBuilder;
// use crate::chip::memory::pointer::raw::RawPointer;
// use crate::chip::memory::time::Time;
// use crate::chip::memory::value::MemoryValue;
// use crate::chip::register::array::ArrayRegister;
// use crate::chip::register::cell::CellType;
// use crate::chip::register::cubic::CubicRegister;
// use crate::chip::register::element::ElementRegister;
// use crate::chip::register::memory::MemorySlice;
// use crate::chip::register::{Register, RegisterSerializable, RegisterSized};

// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// pub struct U16PairRegister(ArrayRegister<ElementRegister>);

// impl RegisterSerializable for U16PairRegister {
//     const CELL: CellType = CellType::U16;

//     fn from_register_unsafe(register: MemorySlice) -> Self {
//         Self(ArrayRegister::from_register_unsafe(register))
//     }

//     fn register(&self) -> &MemorySlice {
//         self.0.register()
//     }
// }

// impl RegisterSized for U16PairRegister {
//     fn size_of() -> usize {
//         2
//     }
// }

// impl Register for U16PairRegister {
//     type Value<T> = [T; 2];

//     fn align<T>(value: &Self::Value<T>) -> &[T] {
//         value
//     }

//     fn value_from_slice<T: Copy>(slice: &[T]) -> Self::Value<T> {
//         [slice[0], slice[1]]
//     }
// }

// impl MemoryValue for U16PairRegister {
//     fn compress<L: crate::chip::AirParameters>(
//         &self,
//         builder: &mut AirBuilder<L>,
//         ptr: RawPointer,
//         time: &Time<L::Field>,
//     ) -> CubicRegister {
//     }
// }
