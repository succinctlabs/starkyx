use super::{AirBuilder, AirParameters};
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cell::CellType;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};

impl<L: AirParameters> AirBuilder<L> {
    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice`.
    pub fn get_local_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Local(self.local_index, size);
        self.local_index += size;
        register
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice` on the
    /// next row.
    fn get_next_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Next(self.next_index, size);
        self.next_index += size;
        register
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice`. Each
    /// cell will be range checked using the lookup table to be in the range `[0, 2^16]`.
    fn get_local_u16_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Local(self.local_arithmetic_index, size);
        self.local_arithmetic_index += size;
        register
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice` in the
    /// next row. Each cell will be range checked using the lookup table to be in the range
    /// `[0, 2^16]`.
    fn get_next_u16_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Next(self.next_arithmetic_index, size);
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
                unimplemented!("Bit registers are not yet supported")
                // let reg = self.get_local_memory(T::size_of());
                // let constraint_expr =
                //     ConstraintExpression::from(reg.expr() * (reg.expr() - F::ONE));
                // self.constraints.push(Constraint::All(constraint_expr));
                // reg
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
                unimplemented!("Bit registers are not yet supported")
                // let reg = self.get_local_memory(size_of);
                // let constraint = Constraint::All((reg.expr() * (reg.expr() - F::ONE)).into());
                // self.constraints.push(constraint);
                // reg
            }
        };
        ArrayRegister::<T>::from_register_unsafe(register)
    }

    /// Allocates a new register on the next row according to type `T` which implements the Register
    /// trait and returns it.
    pub fn alloc_next<T: Register>(&mut self) -> T {
        let register = match T::CELL {
            CellType::Element => self.get_next_memory(T::size_of()),
            CellType::U16 => self.get_next_u16_memory(T::size_of()),
            CellType::Bit => {
                unimplemented!("Bit registers are not yet supported")
                // let reg = self.get_next_memory(T::size_of());
                // let constraint = Constraint::All((reg.expr() * (reg.expr() - F::ONE)).into());
                // self.constraints.push(constraint);
                // reg
            }
        };
        T::from_register(register)
    }

    // /// This method should be applied to any data that needs to be manually written to the trace by
    // /// the user during trace generation. It currently does not do any actual checks, but this can
    // /// be changed later.
    // pub fn write_data<T: Register>(&mut self, data: &T) -> Result<()> {
    //     let register = data.register();
    //     let label = InstructionId::Write(*register);
    //     let existing_value = self
    //         .instruction_indices
    //         .insert(label, self.write_instructions.len());
    //     if existing_value.is_some() {
    //         return Err(anyhow!("Instruction label already exists"));
    //     }
    //     self.write_instructions.push(WriteInstruction(*register));
    //     Ok(())
    // }
}
