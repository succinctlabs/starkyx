pub mod arithmetic;
pub mod memory;
pub mod range_check;

use alloc::collections::BTreeSet;

use anyhow::Result;

use super::constraint::Constraint;
use super::instruction::set::{AirInstruction, InstructionSet};
use super::lookup::Lookup;
use super::register::element::ElementRegister;
use super::register::Register;
use super::{AirParameters, Chip};
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct AirBuilder<L: AirParameters> {
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    challenge_index: usize,
    public_inputs_index: usize,
    pub(crate) instructions: InstructionSet<L>,
    pub(crate) constraints: Vec<Constraint<L>>,
    pub(crate) lookup_data: Vec<Lookup<L::Field, L::CubicParams, 1>>,
    range_table: Option<ElementRegister>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub const fn new() -> Self {
        Self {
            local_index: L::NUM_ARITHMETIC_COLUMNS,
            next_index: L::NUM_ARITHMETIC_COLUMNS,
            local_arithmetic_index: 0,
            next_arithmetic_index: 0,
            challenge_index: 0,
            public_inputs_index: 0,
            instructions: BTreeSet::new(),
            constraints: Vec::new(),
            lookup_data: Vec::new(),
            range_table: None,
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

        // // Add instruction to the instruction list
        // self.instructions
        //     .insert(instruction.clone())
        //     .then_some(())
        //     .ok_or_else(|| {
        //         anyhow!(
        //             "Instruction ID {:?} already exists in the instruction set",
        //             instruction.id()
        //         )
        //     })
        Ok(())
    }

    #[inline]
    pub fn range_fn(element: L::Field) -> usize {
        element.as_canonical_u64() as usize
    }

    pub fn build(mut self) -> (Chip<L>, InstructionSet<L>) {
        let execution_trace_length = self.local_index;
        // Add the range checks
        if L::NUM_ARITHMETIC_COLUMNS > 0 {
            self.arithmetic_range_checks();
        }

        // Check the number of columns in comparison to config
        let num_free_columns = self.local_index - L::NUM_ARITHMETIC_COLUMNS; //self.local_index;
        if num_free_columns > L::NUM_FREE_COLUMNS {
            panic!(
                "Not enough free columns. Expected {} free columns, got {}.",
                num_free_columns,
                L::NUM_FREE_COLUMNS
            );
        } else if num_free_columns < L::NUM_FREE_COLUMNS {
            println!(
                "Warning: {} free columns unused",
                L::NUM_FREE_COLUMNS - num_free_columns
            );
        }
        let num_arithmetic_columns = self.local_arithmetic_index;
        if num_arithmetic_columns > L::NUM_ARITHMETIC_COLUMNS {
            panic!(
                "Not enough arithmetic columns. Expected {} arithmetic columns, got {}.",
                num_arithmetic_columns,
                L::NUM_ARITHMETIC_COLUMNS
            );
        } else if num_arithmetic_columns < L::NUM_ARITHMETIC_COLUMNS {
            println!(
                "Warning: {} arithmetic columns unused",
                L::NUM_ARITHMETIC_COLUMNS - num_arithmetic_columns
            );
        }

        (
            Chip {
                constraints: self.constraints,
                num_challenges: self.challenge_index,
                execution_trace_length,
                lookup_data: self.lookup_data,
                range_table: self.range_table,
            },
            self.instructions,
        )
    }
}

#[cfg(test)]
pub mod tests {
    pub use std::sync::mpsc::channel;

    pub use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::air::fibonacci::FibonacciAir;
    pub use crate::air::parser::{AirParser, TraceWindowParser};
    pub use crate::air::RAir;
    pub use crate::chip::instruction::empty::EmptyInstruction;
    pub use crate::chip::register::element::ElementRegister;
    pub use crate::chip::register::u16::U16Register;
    pub use crate::chip::register::RegisterSerializable;
    pub use crate::chip::trace::generator::ArithmeticGenerator;
    pub use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    pub use crate::maybe_rayon::*;
    pub use crate::plonky2::stark::config::PoseidonGoldilocksStarkConfig;
    pub(crate) use crate::plonky2::stark::tests::{test_recursive_starky, test_starky};
    pub use crate::plonky2::stark::Starky;

    #[derive(Debug, Clone)]
    pub struct FibonacciParameters;

    impl const AirParameters for FibonacciParameters {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;
        type Instruction = EmptyInstruction<GoldilocksField>;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 2;

        fn num_rows_bits() -> usize {
            5
        }
    }

    #[test]
    fn test_builder_fibonacci_air() {
        type F = GoldilocksField;
        type L = FibonacciParameters;

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

        let generator = ArithmeticGenerator::<L>::new(&public_inputs);

        let writer = generator.new_writer();

        writer.write(&x_0, &[F::ZERO], 0);
        writer.write(&x_1, &[F::ONE], 0);

        for i in 0..L::num_rows() {
            writer.write_instruction(&constr_1, i);
            writer.write_instruction(&constr_2, i);
        }
        let trace = generator.trace_clone();

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
        type L = FibonacciParameters;
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

        let generator = ArithmeticGenerator::<L>::new(&public_inputs);

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

    #[derive(Debug, Clone)]
    pub struct SimpleTestParameters;

    impl const AirParameters for SimpleTestParameters {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;
        type Instruction = EmptyInstruction<GoldilocksField>;
        const NUM_ARITHMETIC_COLUMNS: usize = 2;
        const NUM_FREE_COLUMNS: usize = 11;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_builder_simple_range_check() {
        type F = GoldilocksField;
        type L = SimpleTestParameters;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_0 = builder.alloc::<U16Register>();
        let x_1 = builder.alloc::<U16Register>();

        let (air, _) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&[]);

        let (tx, rx) = channel();
        for i in 0..L::num_rows() {
            let writer = generator.new_writer();
            let handle = tx.clone();
            rayon::spawn(move || {
                writer.write(&x_0, &[F::ZERO], i);
                writer.write(&x_1, &[F::from_canonical_usize(i)], i);
                handle.send(1).unwrap();
            });
        }
        drop(tx);
        for msg in rx.iter() {
            assert!(msg == 1);
        }
        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
