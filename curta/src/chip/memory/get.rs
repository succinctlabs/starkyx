use serde::{Deserialize, Serialize};

use super::pointer::raw::RawPointer;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetInstruction {
    ptr: RawPointer,
    register: MemorySlice,
}

impl<AP: AirParser> AirConstraint<AP> for GetInstruction {
    // No constraints for this instruction.
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: Field> Instruction<F> for GetInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let mut memory = writer.memory_mut().unwrap();
        let key = self.ptr.read(writer, row_index);
        let entry = memory.get_mut(&key).unwrap_or_else(|| {
            panic!(
                "Memory uninitialized at: \n
                pointer {:?}]\n
                value {:?} \n
                row_index: {:?}\n",
                key, self.register, row_index
            )
        });
        if entry.multiplicity == F::ZERO {
            panic!(
                "Attempt to read with multiplicity zero at: \n
                pointer {:?}]\n
                value {:?} \n
                row_index: {:?}\n",
                key, self.register, row_index
            )
        }
        entry.multiplicity -= F::ONE;
        writer.write_slice(&self.register, &entry.value, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let key = self.ptr.read_from_air(writer);
        let row_index = writer.row_index();
        let entry = writer.memory_mut().get_mut(&key).unwrap_or_else(|| {
            panic!(
                "Memory uninitialized at: \n
                pointer {:?}]\n
                value {:?} \n
                row_index: {:?}\n",
                key, self.register, row_index
            )
        });
        if entry.multiplicity == F::ZERO {
            panic!(
                "Attempt to read with multiplicity zero at: \n
                pointer {:?}]\n
                value {:?} \n
                row_index: {:?}\n",
                key, self.register, row_index
            )
        }
        entry.multiplicity -= F::ONE;
        let value = entry.value.to_vec();
        writer.write_slice(&self.register, &value);
    }
}

impl GetInstruction {
    pub fn new(ptr: RawPointer, register: MemorySlice) -> Self {
        Self { ptr, register }
    }
}
