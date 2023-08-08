use std::sync::mpsc::Sender;

use super::value::ByteOperation;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::memory::MemorySlice;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct ByteOperationInstruction {
    tx: Sender<ByteOperation<u8>>,
    inner: ByteOperation<ByteRegister>,
    global: bool,
    // filter: ArithmeticExpression<F>,
}

impl ByteOperationInstruction {
    pub fn new(
        tx: Sender<ByteOperation<u8>>,
        inner: ByteOperation<ByteRegister>,
        global: bool,
        // filter: ArithmeticExpression<F>,
    ) -> Self {
        ByteOperationInstruction {
            tx,
            inner,
            global,
            // filter,
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
        if self.global && row_index!= 0 {
            return;
        }
        let value = self.inner.write(writer, row_index);
        self.tx.send(value).unwrap();
    }
}
