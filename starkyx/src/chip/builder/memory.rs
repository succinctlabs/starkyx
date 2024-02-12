use super::{AirBuilder, AirParameters};
use crate::chip::instruction::set::AirInstruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cell::CellType;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};

impl<L: AirParameters> AirBuilder<L> {
    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice`.
    pub(crate) fn get_local_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Local(self.local_index, size);
        self.local_index += size;
        register
    }

    fn get_extended_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Local(self.extended_index, size);
        self.extended_index += size;
        register
    }

    fn get_challenge_memory(&mut self, size: usize) -> MemorySlice {
        self.shared_memory.get_challenge_memory(size)
    }

    fn get_global_memory(&mut self, size: usize) -> MemorySlice {
        self.shared_memory.get_global_memory(size)
    }

    fn get_public_memory(&mut self, size: usize) -> MemorySlice {
        self.shared_memory.get_public_memory(size)
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice`. Each
    /// cell will be range checked using the lookup table to be in the range `[0, 2^16]`.
    fn get_local_u16_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Local(self.local_arithmetic_index, size);
        self.local_arithmetic_index += size;
        register
    }

    /// Allocates a new local register according to type `T` which implements the Register trait
    /// and returns it.
    pub fn alloc<T: Register>(&mut self) -> T {
        let register = match T::CELL {
            CellType::Element => self.get_local_memory(T::size_of()),
            CellType::U16 => self.get_local_u16_memory(T::size_of()),
            CellType::Bit => {
                let reg = self.get_local_memory(T::size_of());
                let constraint = AirInstruction::bits(&reg);
                self.register_air_instruction_internal(constraint);
                reg
            }
        };
        T::from_register(register)
    }

    /// Allocates a new local register according to type `T` which implements the Register trait
    /// and returns it.
    pub(crate) fn alloc_extended<T: Register>(&mut self) -> T {
        let register = match T::CELL {
            CellType::Element => self.get_extended_memory(T::size_of()),
            CellType::U16 => unreachable!("Extended U16 not implemented"),
            CellType::Bit => {
                let reg = self.get_extended_memory(T::size_of());
                let constraint = AirInstruction::bits(&reg);
                self.register_air_instruction_internal(constraint);
                reg
            }
        };
        T::from_register(register)
    }

    pub fn alloc_array<T: Register>(&mut self, length: usize) -> ArrayRegister<T> {
        let size_of = T::size_of() * length;
        let register = match T::CELL {
            CellType::Element => self.get_local_memory(size_of),
            CellType::U16 => self.get_local_u16_memory(size_of),
            CellType::Bit => {
                let reg = self.get_local_memory(size_of);
                let constraint = AirInstruction::bits(&reg);
                self.register_air_instruction_internal(constraint);
                reg
            }
        };
        ArrayRegister::<T>::from_register_unsafe(register)
    }

    pub fn alloc_array_extended<T: Register>(&mut self, length: usize) -> ArrayRegister<T> {
        let size_of = T::size_of() * length;
        let register = match T::CELL {
            CellType::Element => self.get_extended_memory(size_of),
            CellType::U16 => unreachable!("Extended U16 not implemented"),
            CellType::Bit => {
                let reg = self.get_extended_memory(size_of);
                let constraint = AirInstruction::bits(&reg);
                self.register_air_instruction_internal(constraint);
                reg
            }
        };
        ArrayRegister::<T>::from_register_unsafe(register)
    }

    pub fn alloc_challenge<T: Register>(&mut self) -> T {
        let register = self.get_challenge_memory(T::size_of());
        T::from_register(register)
    }

    pub fn alloc_array_challenge<T: Register>(&mut self, length: usize) -> ArrayRegister<T> {
        let size_of = T::size_of() * length;
        let register = self.get_challenge_memory(size_of);
        ArrayRegister::<T>::from_register_unsafe(register)
    }

    /// Allocates a new local register according to type `T` which implements the Register trait
    /// and returns it.
    pub fn alloc_global<T: Register>(&mut self) -> T {
        let register = match T::CELL {
            CellType::Element => self.get_global_memory(T::size_of()),
            CellType::U16 => unreachable!("Global U16 not supported"),
            CellType::Bit => self.get_global_memory(T::size_of()),
        };
        T::from_register(register)
    }

    pub fn alloc_array_global<T: Register>(&mut self, length: usize) -> ArrayRegister<T> {
        let size_of = T::size_of() * length;
        let register = match T::CELL {
            CellType::Element => self.get_global_memory(size_of),
            CellType::U16 => unreachable!("Global U16 not supported"),
            CellType::Bit => self.get_global_memory(size_of),
        };
        ArrayRegister::<T>::from_register_unsafe(register)
    }

    /// Allocates a new local register according to type `T` which implements the Register trait
    /// and returns it.
    pub fn alloc_public<T: Register>(&mut self) -> T {
        let register = match T::CELL {
            CellType::Element => self.get_public_memory(T::size_of()),
            CellType::U16 => {
                let register = self.get_public_memory(T::size_of());
                let elements = ArrayRegister::<ElementRegister>::from_register_unsafe(register);
                for element in elements {
                    self.global_arithmetic.push(element);
                }
                register
            }
            CellType::Bit => self.get_public_memory(T::size_of()),
        };
        T::from_register(register)
    }

    pub fn alloc_array_public<T: Register>(&mut self, length: usize) -> ArrayRegister<T> {
        let size_of = T::size_of() * length;
        let register = match T::CELL {
            CellType::Element => self.get_public_memory(size_of),
            CellType::U16 => {
                let register = self.get_public_memory(size_of);
                let elements = ArrayRegister::<ElementRegister>::from_register_unsafe(register);
                for element in elements {
                    self.global_arithmetic.push(element);
                }
                register
            }
            CellType::Bit => self.get_public_memory(size_of),
        };
        ArrayRegister::<T>::from_register_unsafe(register)
    }

    pub fn is_local<T: RegisterSerializable>(&self, register: &T) -> bool {
        match register.register() {
            MemorySlice::Local(index, _) => {
                *index < L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS
                    && *index >= L::NUM_ARITHMETIC_COLUMNS
            }
            _ => false,
        }
    }

    pub fn is_arithmetic<T: RegisterSerializable>(&self, register: &T) -> bool {
        match register.register() {
            MemorySlice::Local(index, _) => *index < L::NUM_ARITHMETIC_COLUMNS,
            _ => false,
        }
    }

    pub fn is_extended<T: RegisterSerializable>(&self, register: &T) -> bool {
        match register.register() {
            MemorySlice::Local(index, _) => {
                *index >= L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS
            }
            _ => false,
        }
    }
}
