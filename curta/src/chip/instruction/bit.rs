use serde::{Deserialize, Serialize};

use super::ConstraintInstruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::memory::MemorySlice;
use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BitConstraint(pub MemorySlice);

impl ConstraintInstruction for BitConstraint {}

impl<AP: AirParser> AirConstraint<AP> for BitConstraint {
    fn eval(&self, parser: &mut AP) {
        let bit_constraint = |bit: AP::Var, parser: &mut AP| {
            let bit_minus_one = parser.sub_const(bit, AP::Field::ONE);
            let constraint = parser.mul(bit, bit_minus_one);
            parser.constraint(constraint);
        };

        let bits = self.0.eval_slice(parser).to_vec();
        for bit in bits {
            bit_constraint(bit, parser);
        }
    }
}
