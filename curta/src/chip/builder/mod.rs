pub mod arithmetic;
pub mod memory;

use super::constraint::arithmetic::expression::ArithmeticExpression;
use super::constraint::Constraint;
use super::instruction::assign::{AssignInstruction, AssignType};
use super::instruction::node::{InstructionNode, WrappedInstruction};
use super::instruction::set::InstructionSet;
use super::instruction::Instruction;
use super::register::Register;
use super::AirParameters;

#[derive(Debug, Clone)]
pub struct AirBuilder<L: AirParameters> {
    num_rows: usize,
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    instructions: Vec<InstructionSet<L::Field, L::Instruction>>,
    nodes: Vec<InstructionNode<L::Field, L::Instruction>>,
    constraints: Vec<Constraint<L>>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub const fn new(num_rows: usize) -> Self {
        Self {
            num_rows: num_rows,
            local_index: L::NUM_ARITHMETIC_COLUMNS,
            next_index: L::NUM_ARITHMETIC_COLUMNS,
            local_arithmetic_index: 0,
            next_arithmetic_index: 0,
            instructions: Vec::new(),
            nodes: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Adds the ability to write to trace location represented by a data register.
    ///
    /// Registers a write instruction into the builder
    pub fn write_data<T: Register>(&mut self, data: &T) {
        let instruction = InstructionSet::write(data.register());
        self.register_from_instruction_set(instruction.into());
    }

    /// Registers an custom instruction with the builder.
    pub fn register_instruction<I>(&mut self, instruction: I)
    where
        L::Instruction: From<I>,
    {
        let instr = L::Instruction::from(instruction);
        self.register_from_instruction_set(InstructionSet::from(instr));
    }

    /// Register an instrucgtion into the builder.
    fn register_from_instruction_set(
        &mut self,
        instruction: InstructionSet<L::Field, L::Instruction>,
    ) {
        // Add the constraints
        self.constraints
            .push(Constraint::from_instruction_set(instruction.clone()));

        // If the instruction doesn't write to the trace, do nothing, otherwise,
        // add the instruction to the instruction dependency graph.
        if instruction.trace_layout().len() == 0 {
            return;
        }

        // Add the instruction to the dependency graph
        let inputs = instruction.inputs();
        let mut nodes = (0..self.num_rows)
            .map(|row_index| {
                let node = InstructionNode::new((instruction.clone(), row_index).into());
                node
            })
            .collect::<Vec<_>>();

        for other in self.instructions.iter() {
            for output in other.trace_layout() {
                if inputs.contains(&output) {
                    for row_index in 0..self.num_rows {
                        let other_node = WrappedInstruction::new(other.clone(), row_index);
                        nodes[row_index].add_dep(other_node);
                    }
                }
                if inputs.contains(&output.next()) {
                    for row_index in 1..self.num_rows {
                        let other_node = WrappedInstruction::new(other.clone(), row_index);
                        nodes[row_index - 1].add_dep(other_node);
                    }
                }
            }
        }
        // Add instruction to the instruction list
        self.instructions.push(instruction.clone());
        // Add all the nodes to the node list
        self.nodes.append(&mut nodes);
    }

    pub fn set_to_expression<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) {
        let instr = AssignInstruction::new(expression.clone(), *data.register(), AssignType::All);
        self.register_from_instruction_set(InstructionSet::Assign(instr));
    }
}

#[cfg(test)]
mod tests {}
