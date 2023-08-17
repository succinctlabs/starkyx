pub mod arithmetic;
pub mod memory;
pub mod range_check;
pub mod shared_memory;

use core::cmp::Ordering;

use anyhow::Result;

use self::shared_memory::SharedMemory;
use super::arithmetic::expression::ArithmeticExpression;
use super::constraint::Constraint;
use super::instruction::set::AirInstruction;
use super::register::element::ElementRegister;
use super::register::{Register, RegisterSerializable};
use super::table::accumulator::Accumulator;
use super::table::bus::channel::BusChannel;
use super::table::evaluation::Evaluation;
use super::table::lookup::Lookup;
use super::{AirParameters, Chip};
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub struct AirBuilder<L: AirParameters> {
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    extended_index: usize,
    shared_memory: SharedMemory,
    pub(crate) instructions: Vec<AirInstruction<L::Field, L::Instruction>>,
    pub(crate) global_instructions: Vec<AirInstruction<L::Field, L::Instruction>>,
    pub(crate) constraints: Vec<Constraint<L>>,
    pub(crate) global_constraints: Vec<Constraint<L>>,
    pub(crate) accumulators: Vec<Accumulator<L::Field, L::CubicParams>>,
    pub(crate) bus_channels: Vec<BusChannel<L::Field, L::CubicParams>>,
    pub(crate) lookup_data: Vec<Lookup<L::Field, L::CubicParams>>,
    pub(crate) evaluation_data: Vec<Evaluation<L::Field, L::CubicParams>>,
    range_table: Option<ElementRegister>,
}

#[derive(Debug, Clone)]
pub struct AirTraceData<L: AirParameters> {
    pub num_challenges: usize,
    pub num_public_inputs: usize,
    pub num_global_values: usize,
    pub instructions: Vec<AirInstruction<L::Field, L::Instruction>>,
    pub global_instructions: Vec<AirInstruction<L::Field, L::Instruction>>,
    pub accumulators: Vec<Accumulator<L::Field, L::CubicParams>>,
    pub bus_channels: Vec<BusChannel<L::Field, L::CubicParams>>,
    pub lookup_data: Vec<Lookup<L::Field, L::CubicParams>>,
    pub evaluation_data: Vec<Evaluation<L::Field, L::CubicParams>>,
    pub range_table: Option<ElementRegister>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new() -> Self {
        Self::new_with_shared_memory(SharedMemory::new())
    }

    pub fn new_with_shared_memory(shared_memory: SharedMemory) -> Self {
        Self {
            local_index: L::NUM_ARITHMETIC_COLUMNS,
            next_index: L::NUM_ARITHMETIC_COLUMNS,
            local_arithmetic_index: 0,
            next_arithmetic_index: 0,
            extended_index: L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS,
            shared_memory,
            instructions: Vec::new(),
            global_instructions: Vec::new(),
            constraints: Vec::new(),
            global_constraints: Vec::new(),
            accumulators: Vec::new(),
            bus_channels: Vec::new(),
            lookup_data: Vec::new(),
            evaluation_data: Vec::new(),
            range_table: None,
        }
    }

    /// Adds the ability to write to trace location represented by a data register.
    ///
    /// Registers a write instruction into the builder
    pub fn write_data<T: Register>(&mut self, data: &T) {
        let instruction = AirInstruction::write(data.register());
        self.register_air_instruction_internal(instruction).unwrap();
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

    /// Registers an custom instruction with the builder.
    pub fn register_global_instruction<I>(&mut self, instruction: I)
    where
        L::Instruction: From<I>,
    {
        let instr = L::Instruction::from(instruction);
        self.register_global_air_instruction_internal(AirInstruction::from(instr))
            .unwrap();
    }

    /// Registers an custom instruction with the builder.
    pub fn register_instruction_with_filter<I>(
        &mut self,
        instruction: I,
        filter: ArithmeticExpression<L::Field>,
    ) where
        L::Instruction: From<I>,
    {
        let instr = AirInstruction::from(L::Instruction::from(instruction));
        let filtered_instr = instr.as_filtered(filter);

        self.register_air_instruction_internal(filtered_instr)
            .unwrap();
    }

    /// Register an instruction into the builder.
    pub(crate) fn register_air_instruction_internal(
        &mut self,
        instruction: AirInstruction<L::Field, L::Instruction>,
    ) -> Result<()> {
        // Add the instruction to the list
        self.instructions.push(instruction.clone());
        // Add the constraints
        self.constraints
            .push(Constraint::from_instruction_set(instruction));

        Ok(())
    }

    /// Register a global instruction into the builder.
    pub(crate) fn register_global_air_instruction_internal(
        &mut self,
        instruction: AirInstruction<L::Field, L::Instruction>,
    ) -> Result<()> {
        // Add the instruction to the list
        self.global_instructions.push(instruction.clone());
        // Add the constraints
        self.global_constraints
            .push(Constraint::from_instruction_set(instruction));

        Ok(())
    }

    #[inline]
    pub fn range_fn(element: L::Field) -> usize {
        element.as_canonical_u64() as usize
    }

    pub fn clock(&mut self) -> ElementRegister {
        let clk = self.alloc::<ElementRegister>();

        self.set_to_expression_first_row(&clk, ArithmeticExpression::zero());
        self.set_to_expression_transition(&clk.next(), clk.expr() + L::Field::ONE);

        clk
    }

    pub fn build(mut self) -> (Chip<L>, AirTraceData<L>) {
        // constrain all bus channels
        for channel in self.bus_channels.iter() {
            self.constraints.push(channel.clone().into());
        }

        // Add the range checks
        if L::NUM_ARITHMETIC_COLUMNS > 0 {
            self.arithmetic_range_checks();
        }

        // Check the number of columns in comparison to config
        let num_free_columns = self.local_index - L::NUM_ARITHMETIC_COLUMNS;

        match num_free_columns.cmp(&L::NUM_FREE_COLUMNS) {
            Ordering::Greater => panic!(
                "Not enough free columns. Expected {} free columns, got {}.",
                num_free_columns,
                L::NUM_FREE_COLUMNS
            ),
            Ordering::Less => {
                println!(
                    "Warning: {} free columns unused",
                    L::NUM_FREE_COLUMNS - num_free_columns
                );
            }
            Ordering::Equal => {}
        }

        let num_arithmetic_columns = self.local_arithmetic_index;

        match num_arithmetic_columns.cmp(&L::NUM_ARITHMETIC_COLUMNS) {
            Ordering::Greater => panic!(
                "Not enough arithmetic columns. Expected {} arithmetic columns, got {}.",
                num_arithmetic_columns,
                L::NUM_ARITHMETIC_COLUMNS
            ),
            Ordering::Less => {
                println!(
                    "Warning: {} arithmetic columns unused",
                    L::NUM_ARITHMETIC_COLUMNS - num_arithmetic_columns
                );
            }
            Ordering::Equal => {}
        }

        let num_extended_columns =
            self.extended_index - L::NUM_ARITHMETIC_COLUMNS - L::NUM_FREE_COLUMNS;

        match num_extended_columns.cmp(&L::EXTENDED_COLUMNS) {
            Ordering::Greater => panic!(
                "Not enough extended columns. Expected {} extended columns, got {}.",
                num_extended_columns,
                L::EXTENDED_COLUMNS
            ),
            Ordering::Less => {
                println!(
                    "Warning: {} extended columns unused",
                    L::EXTENDED_COLUMNS - num_extended_columns
                );
            }
            Ordering::Equal => {}
        }

        let execution_trace_length = self.local_index;
        (
            Chip {
                constraints: self.constraints,
                global_constraints: self.global_constraints,
                num_challenges: self.shared_memory.challenge_index(),
                execution_trace_length,
                num_public_inputs: self.shared_memory.public_index(),
                num_global_values: self.shared_memory.global_index(),
            },
            AirTraceData {
                num_challenges: self.shared_memory.challenge_index(),
                num_public_inputs: self.shared_memory.public_index(),
                num_global_values: self.shared_memory.global_index(),
                instructions: self.instructions,
                global_instructions: self.global_instructions,
                accumulators: self.accumulators,
                bus_channels: self.bus_channels,
                lookup_data: self.lookup_data,
                evaluation_data: self.evaluation_data,
                range_table: self.range_table,
            },
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    pub use std::sync::mpsc::channel;

    pub use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::air::fibonacci::FibonacciAir;
    pub use crate::air::parser::AirParser;
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
    pub use crate::trace::window_parser::TraceWindowParser;

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

        let (mut air, mut air_data) = builder.build();
        air.num_public_inputs = 3;
        air_data.num_public_inputs = 3;

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(L::num_rows() - 1, F::ZERO, F::ONE),
        ];

        let generator = ArithmeticGenerator::<L>::new(air_data);

        let writer = generator.new_writer();

        writer.write(&x_0, &F::ZERO, 0);
        writer.write(&x_1, &F::ONE, 0);

        for i in 0..L::num_rows() {
            writer.write_instruction(&constr_1, i);
            writer.write_instruction(&constr_2, i);
        }
        let trace = generator.trace_clone();

        for window in trace.windows_iter() {
            assert_eq!(window.local_slice.len(), 2);
            let mut window_parser = TraceWindowParser::new(window, &[], &[], &public_inputs);
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

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(L::num_rows() - 1, F::ZERO, F::ONE),
        ];

        let (mut air, mut air_data) = builder.build();
        air.num_public_inputs = 3;
        air_data.num_public_inputs = 3;

        let generator = ArithmeticGenerator::<L>::new(air_data);

        let writer = generator.new_writer();

        writer.write(&x_0, &F::ZERO, 0);
        writer.write(&x_1, &F::ONE, 0);

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
        const NUM_FREE_COLUMNS: usize = 4;
        const EXTENDED_COLUMNS: usize = 13;

        fn num_rows_bits() -> usize {
            14
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

        let clk = builder.clock();
        let clk_expected = builder.alloc::<ElementRegister>();

        builder.assert_equal(&clk, &clk_expected);

        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data);

        let (tx, rx) = channel();
        for i in 0..L::num_rows() {
            let writer = generator.new_writer();
            let handle = tx.clone();
            // rayon::spawn(move || {
            writer.write(&x_0, &F::ZERO, i);
            writer.write(&x_1, &F::from_canonical_usize(0), i);
            writer.write(&clk_expected, &F::from_canonical_usize(i), i);
            writer.write_row_instructions(&generator.air_data, i);
            handle.send(1).unwrap();
            // });
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
