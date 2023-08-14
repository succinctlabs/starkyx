use alloc::sync::Arc;
use std::sync::Mutex;

use super::value::ByteOperation;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::lookup_table::multiplicity_data::MultiplicityData;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct ByteOperationInstruction {
    multiplicity_data: Arc<Mutex<MultiplicityData>>,
    inner: ByteOperation<ByteRegister>,
    global: bool,
}

impl ByteOperationInstruction {
    pub fn new(
        multiplicity_data: Arc<Mutex<MultiplicityData>>,
        inner: ByteOperation<ByteRegister>,
        global: bool,
    ) -> Self {
        ByteOperationInstruction {
            multiplicity_data,
            inner,
            global,
        }
    }
}

impl<AP: AirParser> AirConstraint<AP> for ByteOperationInstruction {
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: PrimeField64> Instruction<F> for ByteOperationInstruction {
    fn inputs(&self) -> Vec<MemorySlice> {
        self.inner.inputs()
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        self.inner.trace_layout()
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        if self.global && row_index != 0 {
            return;
        }
        let value = self.inner.write(writer, row_index);
        self.multiplicity_data.lock().unwrap().update(&value);
    }
}
