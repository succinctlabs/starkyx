use log::trace;
use serde::{Deserialize, Serialize};

use super::instruction::{MemoryOutput, MemorySliceIndex};
use super::map::MemEntry;
use super::pointer::raw::RawPointer;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetInstruction<F> {
    ptr: RawPointer,
    register: MemorySlice,
    multiplicity: Option<ElementRegister>,
    memory_output: Option<MemoryOutput<F>>,
}

impl<AP: AirParser> AirConstraint<AP> for SetInstruction<AP::Field> {
    // No constraints for this instruction.
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: Field> Instruction<F> for SetInstruction<F> {
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

        if let Some(memory_output) = &self.memory_output {
            // Can assume that self.write_ts and self.index are not None.

            let mult = if let Some(multiplicity) = self.multiplicity {
                writer.read(&multiplicity, row_index)
            } else {
                F::ONE
            };

            let index = match memory_output.index {
                Some(MemorySliceIndex::Index(index)) => Some(F::from_canonical_usize(index)),
                Some(MemorySliceIndex::IndexElement(index)) => Some(writer.read(&index, row_index)),
                None => None,
            };

            trace!(
                    "memory set - row: {:?}, mem label: {:?}, index: {:?}, multiplicity: {:?}, ts: {:?}",
                    row_index,
                    memory_output.label,
                    index,
                    mult,
                    writer.read_expression(&memory_output.ts.0, row_index)[0],
                );
        }

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

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let row_index = writer.row_index();
        let multiplicity = if let Some(mult) = self.multiplicity {
            writer.read(&mult)
        } else {
            F::ONE
        };

        if let Some(memory_output) = &self.memory_output {
            let mult = if let Some(multiplicity) = self.multiplicity {
                writer.read(&multiplicity)
            } else {
                F::ONE
            };

            let index = match memory_output.index {
                Some(MemorySliceIndex::Index(index)) => Some(F::from_canonical_usize(index)),
                Some(MemorySliceIndex::IndexElement(index)) => Some(writer.read(&index)),
                None => None,
            };

            trace!(
                "memory set - row: {:?}, mem label: {}, index: {:?}, multiplicity: {:?}, ts: {:?}",
                row_index,
                memory_output.label,
                index,
                mult,
                writer.read_expression(&memory_output.ts.0)[0],
            );
        }

        let key = self.ptr.read_from_air(writer);

        let value = writer.read_slice(&self.register).to_vec();

        writer
            .memory_mut()
            .0
            .entry(key)
            .and_modify(|v| {
                v.value.copy_from_slice(&value);
                v.multiplicity += multiplicity;
            })
            .or_insert_with(|| MemEntry {
                value,
                multiplicity,
            });
    }
}

impl<F: Field> SetInstruction<F> {
    pub fn new(
        ptr: RawPointer,
        register: MemorySlice,
        multiplicity: Option<ElementRegister>,
        memory_output: Option<MemoryOutput<F>>,
    ) -> Self {
        Self {
            ptr,
            register,
            multiplicity,
            memory_output,
        }
    }
}
