use plonky2::field::extension::Extendable;
use plonky2::{field::extension::FieldExtension, hash::hash_types::RichField};
use plonky2::field::packed::PackedField;
use plonky2::iop::ext_target::ExtensionTarget;

use core::marker::PhantomData;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

use super::circuit::{Register, DataRegister, WitnessRegister, CellType};
use super::instruction::Instruction;



#[derive(Debug, Clone)]
pub struct ChipBuilder<F: RichField + Extendable<D>, const D: usize, C> {
    local_index : usize,
    next_index : usize,
    instructions : Vec<C>,
    range_checks : Vec<usize>,
    initial_index : usize,
    _marker : std::marker::PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize, C : Instruction<F, D> > ChipBuilder<F, D, C> {

   fn new() -> Self {
    Self { local_index: 0, next_index: 0, instructions: vec![], range_checks: vec![], initial_index: 0, _marker: PhantomData }
   } 
    
    /// Adds a new local (i.e. current row) data register of type T
    pub fn add_local_data<T : DataRegister>(&mut self) -> T {
        let length = T::size_of();
        let register = self.get_local_mem(length);
        T::from_raw_register(register)
    }

    pub fn add_next_data<T : DataRegister>(&mut self) -> T {
        let length = T::size_of();
        let register = self.get_next_mem(length);

        match T::CELL {
            CellType::U16 => {
                self.range_checks.push(register.index());
            },
            CellType::Bit => unimplemented!("Bit cells are not yet supported")
        }
        T::from_raw_register(register)
    }

    fn get_local_mem(&mut self, length : usize) -> Register {
        let register = Register::Local(self.local_index, length);
        self.local_index += length;
        register
    } 

    fn get_next_mem(&mut self, length : usize) -> Register {
        let register = Register::Next(self.next_index, length);
        self.next_index += length;
        register
    } 

    pub fn add_instruction(&mut self, mut instruction : C) {
        let witness = self.get_local_mem(instruction.witness_size());
        instruction.set_witness(witness);
        self.instructions.push(instruction);
    }
}