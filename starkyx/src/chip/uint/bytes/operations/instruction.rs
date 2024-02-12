use serde::{Deserialize, Serialize};

use super::value::ByteOperation;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::uint::bytes::register::ByteRegister;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteOperationInstruction {
    inner: ByteOperation<ByteRegister>,
    global: bool,
}

impl ByteOperationInstruction {
    pub fn new(inner: ByteOperation<ByteRegister>, global: bool) -> Self {
        ByteOperationInstruction { inner, global }
    }
}

impl<AP: AirParser> AirConstraint<AP> for ByteOperationInstruction {
    fn eval(&self, _parser: &mut AP) {}
}

impl<F: PrimeField64> Instruction<F> for ByteOperationInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        if self.global && row_index != 0 {
            return;
        }
        self.inner.write(writer, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        if let Some(r) = writer.row_index() {
            if self.global && r != 0 {
                return;
            }
        }
        self.inner.write_to_air(writer);
    }
}
