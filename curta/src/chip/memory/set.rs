use serde::{Deserialize, Serialize};

use super::map::MemEntry;
use super::pointer::raw::RawPointer;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetInstruction {
    ptr: RawPointer,
    register: MemorySlice,
    multiplicity: Option<ElementRegister>,
}

impl<AP: AirParser> AirConstraint<AP> for SetInstruction {
    // No constraints for this instruction.
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: Field> Instruction<F> for SetInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let multiplicity = if let Some(mult) = self.multiplicity {
            writer.read(&mult, row_index)
        } else {
            F::ONE
        };
        let mut memory = writer.memory_mut().unwrap();
        // let value = memory.get_mut(&self.ptr).expect("Memory not initialized.");

        let (id_0, id_1) = self.register.get_range();

        let key = self.ptr.read(writer, row_index);

        match self.register {
            MemorySlice::Local(_, _) => {
                let trace = writer.read_trace().unwrap();
                let slice = &trace.row(row_index)[id_0..id_1];
                memory
                    .0
                    .entry(key)
                    .and_modify(|v| {
                        v.value.copy_from_slice(slice);
                        v.multiplicity += multiplicity;
                    })
                    .or_insert_with(|| MemEntry {
                        value: slice.to_vec(),
                        multiplicity,
                    });
            }
            MemorySlice::Public(_, _) => {
                let public = writer.public().unwrap();
                let slice = &public[id_0..id_1];
                memory
                    .0
                    .entry(key)
                    .and_modify(|v| {
                        v.value.copy_from_slice(slice);
                        v.multiplicity += multiplicity;
                    })
                    .or_insert_with(|| MemEntry {
                        value: slice.to_vec(),
                        multiplicity,
                    });
            }
            MemorySlice::Global(_, _) => {
                let global = writer.global().unwrap();
                let slice = &global[id_0..id_1];
                memory
                    .0
                    .entry(key)
                    .and_modify(|v| {
                        v.value.copy_from_slice(slice);
                        v.multiplicity += multiplicity;
                    })
                    .or_insert_with(|| MemEntry {
                        value: slice.to_vec(),
                        multiplicity,
                    });
            }
            _ => unimplemented!("Cannot write to this memory slice type."),
        };
    }
}

impl SetInstruction {
    pub fn new(
        ptr: RawPointer,
        register: MemorySlice,
        multiplicity: Option<ElementRegister>,
    ) -> Self {
        Self {
            ptr,
            register,
            multiplicity,
        }
    }
}
