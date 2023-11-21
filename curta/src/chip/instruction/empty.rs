use serde::{Deserialize, Serialize};

use super::Instruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::prelude::*;

/// A defult instruction set that contains no custom instructions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmptyInstruction<F> {
    _marker: core::marker::PhantomData<F>,
}

impl<F: Field> Instruction<F> for EmptyInstruction<F> {
    fn write(&self, _writer: &TraceWriter<F>, _row_index: usize) {}

    fn write_to_air(&self, _writer: &mut impl AirWriter<Field = F>) {}
}

impl<F: Field, AP: AirParser<Field = F>> AirConstraint<AP> for EmptyInstruction<F> {
    fn eval(&self, _parser: &mut AP) {}
}
