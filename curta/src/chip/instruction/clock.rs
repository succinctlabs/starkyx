use serde::{Deserialize, Serialize};

use super::Instruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockInstruction {
    pub(crate) clk: ElementRegister,
}

impl<AP: AirParser> AirConstraint<AP> for ClockInstruction {
    fn eval(&self, parser: &mut AP) {
        let clk = self.clk.eval(parser);
        let clk_next = self.clk.next().eval(parser);

        parser.constraint_first_row(clk);

        let mut transition = parser.sub(clk_next, clk);
        transition = parser.sub_const(transition, AP::Field::ONE);
        parser.constraint_transition(transition);
    }
}

impl<F: Field> Instruction<F> for ClockInstruction {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let value = F::from_canonical_usize(row_index);
        writer.write(&self.clk, &value, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let value = F::from_canonical_usize(writer.row_index().unwrap());
        writer.write(&self.clk, &value);
    }
}
