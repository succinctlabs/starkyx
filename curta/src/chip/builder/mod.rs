pub mod arithmetic;
pub mod memory;

use alloc::collections::BTreeSet;

use anyhow::{anyhow, Result};

use super::constraint::Constraint;
use super::instruction::set::{AirInstruction, InstructionSet};
use super::register::Register;
use super::{AirParameters, Chip};
use crate::chip::instruction::Instruction;

#[derive(Debug, Clone)]
pub struct AirBuilder<L: AirParameters> {
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    instructions: InstructionSet<L>,
    constraints: Vec<Constraint<L>>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub const fn new() -> Self {
        Self {
            local_index: L::NUM_ARITHMETIC_COLUMNS,
            next_index: L::NUM_ARITHMETIC_COLUMNS,
            local_arithmetic_index: 0,
            next_arithmetic_index: 0,
            instructions: BTreeSet::new(),
            constraints: Vec::new(),
        }
    }

    /// Adds the ability to write to trace location represented by a data register.
    ///
    /// Registers a write instruction into the builder
    pub fn write_data<T: Register>(&mut self, data: &T) {
        let instruction = AirInstruction::write(data.register());
        self.register_air_instruction_internal(instruction.into())
            .unwrap();
    }

    /// Registers an custom instruction with the builder.
    pub fn register_instruction<I>(&mut self, instruction: I)
    where
        L::Instruction: From<I>,
    {
        let instr = L::Instruction::from(instruction);
        self.register_air_instruction_internal(AirInstruction::from(instr))
            .unwrap();
    }

    /// Register an instrucgtion into the builder.
    fn register_air_instruction_internal(
        &mut self,
        instruction: AirInstruction<L::Field, L::Instruction>,
    ) -> Result<()> {
        // Add the constraints
        self.constraints
            .push(Constraint::from_instruction_set(instruction.clone()));

        // Add instruction to the instruction list
        self.instructions
            .insert(instruction.clone())
            .then_some(())
            .ok_or_else(|| {
                anyhow!(
                    "Instruction ID {:?} already exists in the instruction set",
                    instruction.id()
                )
            })
    }

    pub fn build(self) -> (Chip<L>, InstructionSet<L>) {
        (
            Chip {
                constraints: self.constraints,
            },
            self.instructions,
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
    use crate::chip::trace::generator::ArithmeticGenerator;
    use crate::math::prelude::*;
    use crate::plonky2::stark::config::PoseidonGoldilocksStarkConfig;
    use crate::plonky2::stark::tests::{test_recursive_starky, test_starky};
    use crate::plonky2::stark::Starky;

    #[derive(Debug, Clone)]
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

        let mut builder = AirBuilder::<L>::new();
        let x_0 = builder.alloc::<ElementRegister>();
        let x_1 = builder.alloc::<ElementRegister>();

        // x0' <- x1
        let constr_1 = builder.set_to_expression_transition(&x_0.next(), x_1.expr());
        // x1' <- x0 + x1
        let constr_2 = builder.set_to_expression_transition(&x_1.next(), x_0.expr() + x_1.expr());

        let (air, _) = builder.build();

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(L::num_rows() - 1, F::ZERO, F::ONE),
        ];

        let generator = ArithmeticGenerator::<L>::new();

        let writer = generator.new_writer();

        writer.write(&x_0, &[F::ZERO], 0);
        writer.write(&x_1, &[F::ONE], 0);

        for i in 0..L::num_rows() {
            writer.write_instruction(&constr_1, i);
            writer.write_instruction(&constr_2, i);
        }
        let trace = generator.trace();

        for window in trace.windows_iter() {
            assert_eq!(window.local_slice.len(), 2);
            let mut window_parser = TraceWindowParser::new(window, &[], &public_inputs);
            assert_eq!(window_parser.local_slice().len(), 2);
            air.eval(&mut window_parser);
        }
    }

    #[test]
    fn test_builder_fibonacci_stark() {
        type F = GoldilocksField;
        type L = FibonacciParameters<F>;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_0 = builder.alloc::<ElementRegister>();
        let x_1 = builder.alloc::<ElementRegister>();

        // x0' <- x1
        let constr_1 = builder.set_to_expression_transition(&x_0.next(), x_1.expr());
        // x1' <- x0 + x1
        let constr_2 = builder.set_to_expression_transition(&x_1.next(), x_0.expr() + x_1.expr());

        let (air, _) = builder.build();

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(L::num_rows() - 1, F::ZERO, F::ONE),
        ];

        let generator = ArithmeticGenerator::<L>::new();

        let writer = generator.new_writer();

        writer.write(&x_0, &[F::ZERO], 0);
        writer.write(&x_1, &[F::ONE], 0);

        for i in 0..L::num_rows() {
            writer.write_instruction(&constr_1, i);
            writer.write_instruction(&constr_2, i);
        }
        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
