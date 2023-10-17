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
pub struct GetInstruction {
    ptr: RawPointer,
    register: MemorySlice,
    time: TimeRegister,
}

impl<AP: AirParser> AirConstraint<AP> for GetInstruction {
    // No constraints for this instruction.
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: Field> Instruction<F> for GetInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let memory = writer.memory().unwrap();
        let (value, ts) = memory.get(&self.ptr).expect("Memory not initialized.");
        writer.write_slice(&self.register, value, row_index);
        writer.write(&self.time, ts, row_index);
    }
}

impl GetInstruction {
    pub fn new(ptr: RawPointer, register: MemorySlice, time: TimeRegister) -> Self {
        Self {
            ptr,
            register,
            time,
        }
    }
}
