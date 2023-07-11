pub mod arithmetic;
pub mod memory;

use super::constraint::Constraint;
use super::instruction::set::InstructionSet;
use super::register::Register;
use super::{AirParameters, Chip};

#[derive(Debug, Clone)]
pub struct AirBuilder<L: AirParameters> {
    num_rows: usize,
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    instructions: Vec<InstructionSet<L::Field, L::Instruction>>,
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

        // Add instruction to the instruction list
        self.instructions.push(instruction.clone());
    }

    pub fn build(self) -> (Chip<L>, ()) {
        // Create the instruction graph
        for instruction in self.instructions.iter() {}
        (
            Chip {
                constraints: self.constraints,
            },
            (),
        )
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::air::fibonacci::FibonacciAir;
    use crate::air::parser::{AirParser, TraceWindowParser};
    use crate::air::RAir;
    use crate::chip::instruction::empty::EmptyInstruction;
    use crate::chip::register::element::ElementRegister;
    use crate::chip::register::RegisterSerializable;
    use crate::math::prelude::*;

    pub struct FibonacciParameters<F>(core::marker::PhantomData<F>);

    impl<F: Field> const AirParameters for FibonacciParameters<F> {
        type Field = F;
        type Challenge = F;
        type Instruction = EmptyInstruction<F>;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 2;

        fn num_rows_bits() -> usize {
            5
        }
    }

    #[test]
    fn test_builder_fibonacci_air() {
        type F = GoldilocksField;
        type L = FibonacciParameters<F>;

        let num_rows = 1 << 5 as usize;

        let mut builder = AirBuilder::<L>::new(num_rows);
        let x_0 = builder.alloc::<ElementRegister>();
        let x_1 = builder.alloc::<ElementRegister>();

        // x0' <- x1
        let first_col_constraint = x_0.next().expr() - x_1.expr();
        builder.assert_expression_zero_transition(first_col_constraint);
        // x1' <- x0 + x1
        let second_col_constraint = x_1.next().expr() - (x_0.expr() + x_1.expr());
        builder.assert_expression_zero_transition(second_col_constraint);

        let (air, _) = builder.build();

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(num_rows - 1, F::ZERO, F::ONE),
        ];

        let trace = FibonacciAir::generate_trace(F::ZERO, F::ONE, num_rows);

        for window in trace.windows_iter() {
            assert_eq!(window.local_slice.len(), 2);
            let mut window_parser = TraceWindowParser::new(window, &[], &public_inputs);
            assert_eq!(window_parser.local_slice().len(), 2);
            air.eval(&mut window_parser);
        }
    }
}
