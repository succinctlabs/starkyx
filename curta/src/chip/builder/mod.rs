use dep_graph::DepGraph;

use super::instruction::{Instruction, InstructionId};
use crate::math::prelude::*;

pub struct AirBuilder<F: Field> {
    pub instructions: Vec<Box<dyn Instruction<F>>>,
    pub dep_graph: DepGraph<InstructionId>,
}
