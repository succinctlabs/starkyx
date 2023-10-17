use serde::{Deserialize, Serialize};

use super::pointer::RawPointer;
use super::time::TimeRegister;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetInstruction {
    ptr: RawPointer,
    register: MemorySlice,
    time: TimeRegister,
}

impl<AP: AirParser> AirConstraint<AP> for SetInstruction {
    // No constraints for this instruction.
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: Field> Instruction<F> for SetInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let timestamp = writer.read(&self.time, row_index);

        let mut memory = writer.memory_mut().unwrap();
        let (value, ts) = memory.get_mut(&self.ptr).expect("Memory not initialized.");
        *ts = timestamp;

        let (id_0, id_1) = self.register.get_range();
        match self.register {
            MemorySlice::Local(_, _) => {
                let trace = writer.read_trace().unwrap();
                value.copy_from_slice(&trace.row(row_index)[id_0..id_1]);
            }
            MemorySlice::Public(_, _) => {
                let public = writer.public().unwrap();
                value.copy_from_slice(&public[id_0..id_1]);
            }
            MemorySlice::Global(_, _) => {
                let global = writer.global().unwrap();
                value.copy_from_slice(&global[id_0..id_1]);
            }
            _ => unimplemented!("Cannot write to this memory slice type."),
        }
    }
}

impl SetInstruction {
    pub fn new(ptr: RawPointer, register: MemorySlice, time: TimeRegister) -> Self {
        Self {
            ptr,
            register,
            time,
        }
    }
}
