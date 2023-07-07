pub mod arithmetic;
pub mod memory;

use anyhow::{anyhow, Result};

use super::instruction::node::InstructionNode;
use super::instruction::write::WriteInstruction;
use super::register::Register;
use super::AirParameters;

pub struct FibonacciCircuit;

#[derive(Debug, Clone)]
pub struct AirBuilder<L: AirParameters> {
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    instructions: Vec<InstructionNode<L::Field, L::Instruction>>,
    write_instructions: Vec<WriteInstruction>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub const fn new() -> Self {
        Self {
            local_index: L::NUM_ARITHMETIC_COLUMNS,
            next_index: L::NUM_ARITHMETIC_COLUMNS,
            local_arithmetic_index: 0,
            next_arithmetic_index: 0,
            instructions: Vec::new(),
            write_instructions: Vec::new(),
        }
    }

    pub fn write_data<T: Register>(&mut self, data: &T) {
        let instr = WriteInstruction(*data.register());
        self.write_instructions.push(instr);
    }

    pub fn register_instruction<T>(&mut self, instruction: T) -> Result<()>
    where
        L::Instruction: From<T>,
    {
        let instr = L::Instruction::from(instruction);
        let mut node = InstructionNode::new(instr.into());

        let inputs = node.id().inputs();

        for other_node in self.instructions.iter() {
            let other = other_node.id();
            if node.id() == other_node.id() {
                return Err(anyhow!(
                    "Instruction {:?} already exists in the instruction graph",
                    node
                ));
            }
            for output in other.trace_layout() {
                if inputs.contains(&output) {
                    node.add_dep(other.clone());
                    break;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {}
