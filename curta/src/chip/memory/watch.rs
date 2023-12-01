use log::debug;
use serde::{Deserialize, Serialize};

use super::pointer::raw::RawPointer;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchInstruction {
    ptr: RawPointer,
    name: String,
}

impl<AP: AirParser> AirConstraint<AP> for WatchInstruction {
    // No constraints for this instruction.
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: Field> Instruction<F> for WatchInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let mut memory = writer.memory_mut().unwrap();
        let key = self.ptr.read(writer, row_index);
        let entry_option = memory.get_mut(&key);

        if let Some(entry) = entry_option {
            debug!(
                "row {}: , {}: value: {:?}, multiplicities: {:?}",
                row_index, self.name, entry.value, entry.multiplicity
            );
        } else {
            debug!(
                "row {:?}: , {}: value: Uninitialized, multiplicities: Uninitialized",
                row_index, self.name,
            );
        }
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let key = self.ptr.read_from_air(writer);
        let row_num = if let Some(row_index) = writer.row_index() {
            row_index.to_string()
        } else {
            "None".to_string()
        };
        let entry_option = writer.memory_mut().get_mut(&key);

        if let Some(entry) = entry_option {
            debug!(
                "row {}: , {}: value: {:?}, multiplicities: {:?}",
                row_num, self.name, entry.value, entry.multiplicity
            );
        } else {
            debug!(
                "row {}: , {}: value: Uninitialized, multiplicities: Uninitialized",
                row_num, self.name,
            )
        };
    }
}

impl WatchInstruction {
    pub fn new(ptr: RawPointer, name: String) -> Self {
        Self { ptr, name }
    }
}
