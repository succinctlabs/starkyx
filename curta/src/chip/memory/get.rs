use log::trace;
use serde::{Deserialize, Serialize};

use super::instruction::MemoryOutput;
use super::pointer::raw::RawPointer;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::memory::instruction::MemorySliceIndex;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetInstruction<F> {
    ptr: RawPointer,
    register: MemorySlice,
    memory_output: Option<MemoryOutput<F>>,
}

impl<AP: AirParser> AirConstraint<AP> for GetInstruction<AP::Field> {
    // No constraints for this instruction.
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: Field> Instruction<F> for GetInstruction<F> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let mut memory = writer.memory_mut().unwrap();
        let key = self.ptr.read(writer, row_index);
        let (label, index, write_ts) = if self.memory_output.is_some() {
            let label = &self.memory_output.as_ref().unwrap().label;
            let index = match self.memory_output.as_ref().unwrap().index {
                Some(MemorySliceIndex::Index(index)) => Some(F::from_canonical_usize(index)),
                Some(MemorySliceIndex::IndexElement(index)) => Some(writer.read(&index, row_index)),
                None => None,
            };
            let ts =
                writer.read_expression(&self.memory_output.as_ref().unwrap().ts.0, row_index)[0];
            (Some(label), index, ts)
        } else {
            (None, None, F::ZERO)
        };
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
                memory_label {:?}\n
                index {:?}\n
                value {:?} \n
                row_index: {:?}\n",
                key, label, index, self.register, row_index
            )
        }
        entry.multiplicity -= F::ONE;

        if let Some(memory_output) = &self.memory_output {
            trace!(
                "memory get - row: {:?}, mem label: {}, index: {:?}, multiplicity: {:?}, ts: {:?}",
                row_index,
                memory_output.label,
                index,
                entry.multiplicity,
                write_ts
            );
        }

        writer.write_slice(&self.register, &entry.value, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let key = self.ptr.read_from_air(writer);
        let row_index = writer.row_index();
        let (label, index, write_ts) = if self.memory_output.is_some() {
            let label = &self.memory_output.as_ref().unwrap().label;
            let index = match self.memory_output.as_ref().unwrap().index {
                Some(MemorySliceIndex::Index(index)) => Some(F::from_canonical_usize(index)),
                Some(MemorySliceIndex::IndexElement(index)) => Some(writer.read(&index)),
                None => None,
            };
            let ts = writer.read_expression(&self.memory_output.as_ref().unwrap().ts.0)[0];
            (Some(label), index, ts)
        } else {
            (None, None, F::ZERO)
        };

        let entry = writer.memory_mut().get_mut(&key).unwrap_or_else(|| {
            panic!(
                "Memory uninitialized at: \n
                pointer {:?}]\n
                memory_label {:?}\n
                index {:?}\n
                value {:?} \n
                row_index: {:?}\n",
                key, label, index, self.register, row_index
            )
        });
        if entry.multiplicity == F::ZERO {
            println!("self.memory_output: {:?}", self.memory_output);
            panic!(
                "Attempt to read with multiplicity zero at: \n
                pointer {:?}]\n
                memory_label {:?}\n
                index {:?}\n
                value {:?}\n
                row_index: {:?}\n",
                key, label, index, self.register, row_index
            )
        }
        entry.multiplicity -= F::ONE;
        let value = entry.value.to_vec();

        if self.memory_output.is_some() {
            trace!(
                "memory get - row: {:?}, mem label: {}, index: {:?}, multiplicity: {:?}, ts: {:?}",
                row_index,
                label.unwrap(),
                index,
                entry.multiplicity,
                write_ts
            );
        }

        writer.write_slice(&self.register, &value);
    }
}

impl<F: Field> GetInstruction<F> {
    pub fn new(
        ptr: RawPointer,
        register: MemorySlice,
        memory_output: Option<MemoryOutput<F>>,
    ) -> Self {
        Self {
            ptr,
            register,
            memory_output,
        }
    }
}
