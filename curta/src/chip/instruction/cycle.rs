use serde::{Deserialize, Serialize};

use super::set::AirInstruction;
use super::Instruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::builder::AirBuilder;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cycle<F> {
    pub start_bit: BitRegister,
    pub end_bit: BitRegister,
    num_cycles: usize,
    element: ElementRegister,
    pub start_counter: ElementRegister,
    pub end_counter: ElementRegister,
    group: Vec<F>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn cycle(&mut self, length_log: usize) -> Cycle<L::Field> {
        let start_bit = self.alloc::<BitRegister>();
        let end_bit = self.alloc::<BitRegister>();
        let element = self.alloc::<ElementRegister>();
        let start_counter = self.alloc::<ElementRegister>();
        let end_counter = self.alloc::<ElementRegister>();
        let group = L::Field::two_adic_subgroup(length_log);
        let num_rows = L::num_rows();
        let num_cycles = num_rows / group.len();
        assert_eq!(
            num_rows % group.len(),
            0,
            "Number of rows must be divisible by the size of the group"
        );
        let cycle = Cycle {
            start_bit,
            end_bit,
            num_cycles,
            element,
            start_counter,
            end_counter,
            group,
        };

        self.register_air_instruction_internal(AirInstruction::cycle(cycle.clone()))
            .unwrap();

        cycle
    }
}

impl<AP: AirParser<Field = F>, F: Field> AirConstraint<AP> for Cycle<F> {
    fn eval(&self, parser: &mut AP) {
        // Impose first row constraints
        let element = self.element.eval(parser);
        let element_minus_one = parser.sub_const(element, F::ONE);
        parser.constraint_first_row(element_minus_one);

        // Impose group transition constraints
        let generator = self.group[1];
        let element_next = self.element.next().eval(parser);
        let element_times_generator = parser.mul_const(element, generator);
        let group_constraint = parser.sub(element_next, element_times_generator);
        parser.constraint_transition(group_constraint);

        // Impose compatibility of the bit and the group so that
        // start_bit = 1 <=> element == 1, end_bit = 1 <=> element == gen_inv
        // Impose start_bit * (element - 1) = 0, this implies that start_bit = 1 => element = 1
        let start_bit = self.start_bit.eval(parser);
        let elem_minus_one = parser.sub_const(element, F::ONE);
        let start_bit_constraint = parser.mul(start_bit, elem_minus_one);
        parser.constraint(start_bit_constraint);

        let count = F::from_canonical_usize(self.num_cycles);
        // Impose start_counter constraints, to verify that start_bit = 1 every time element = 1
        let start_counter = self.start_counter.eval(parser);
        parser.constraint_first_row(start_counter);
        let start_counter_next = self.start_counter.next().eval(parser);
        let start_counter_diff = parser.sub(start_counter_next, start_counter);
        let start_counter_constraint = parser.sub(start_counter_diff, start_bit);
        parser.constraint_transition(start_counter_constraint);

        let start_plus_bit = parser.add(start_counter, start_bit);
        let start_counter_last_constraint = parser.sub_const(start_plus_bit, count);
        parser.constraint_last_row(start_counter_last_constraint);

        // Impose end_bit * (element - 1) = 0, this implies that end_bit = 1 => element = gen_inv
        let generator_inv = self.group.last().unwrap();
        let end_bit = self.end_bit.eval(parser);
        let elem_minus_one = parser.sub_const(element, *generator_inv);
        let end_bit_constraint = parser.mul(end_bit, elem_minus_one);
        parser.constraint(end_bit_constraint);

        // Impose end_counter constraints, to verify that end_bit = 1 every time element = gen_inv
        let end_counter = self.end_counter.eval(parser);
        parser.constraint_first_row(end_counter);
        let end_counter_next = self.end_counter.next().eval(parser);
        let end_counter_diff = parser.sub(end_counter_next, end_counter);
        let end_counter_constraint = parser.sub(end_counter_diff, end_bit);
        parser.constraint_transition(end_counter_constraint);

        let end_plus_bit = parser.add(end_counter, end_bit);
        let end_counter_last_constraint = parser.sub_const(end_plus_bit, count);
        parser.constraint_last_row(end_counter_last_constraint);
    }
}

impl<F: Field> Instruction<F> for Cycle<F> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.start_bit.register(),
            *self.end_bit.register(),
            *self.element.register(),
            *self.start_counter.register(),
            *self.end_counter.register(),
        ]
    }

    fn inputs(&self) -> Vec<MemorySlice> {
        Vec::new()
    }

    fn constraint_degree(&self) -> usize {
        2
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let cycle = row_index % self.group.len();
        writer.write(&self.element, &self.group[cycle], row_index);
        let counter = F::from_canonical_usize(row_index / self.group.len());

        if cycle == 0 {
            writer.write(&self.start_bit, &F::ONE, row_index);
            writer.write(&self.end_bit, &F::ZERO, row_index);
            writer.write(&self.start_counter, &(counter), row_index);
            writer.write(&self.end_counter, &counter, row_index);
        } else if cycle == self.group.len() - 1 {
            writer.write(&self.start_bit, &F::ZERO, row_index);
            writer.write(&self.end_bit, &F::ONE, row_index);
            writer.write(&self.start_counter, &(counter + F::ONE), row_index);
            writer.write(&self.end_counter, &counter, row_index);
        } else {
            writer.write(&self.start_bit, &F::ZERO, row_index);
            writer.write(&self.end_bit, &F::ZERO, row_index);
            writer.write(&self.start_counter, &(counter + F::ONE), row_index);
            writer.write(&self.end_counter, &counter, row_index);
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::trace::window_parser::TraceWindowParser;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct CycleTest;

    impl AirParameters for CycleTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = EmptyInstruction<GoldilocksField>;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 5;

        fn num_rows_bits() -> usize {
            8
        }
    }

    #[test]
    fn test_cycle_instruction() {
        type L = CycleTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let cycle = builder.cycle(4);

        let (air, trace_data) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(trace_data);
        let (tx, rx) = channel();
        for i in 0..L::num_rows() {
            let writer = generator.new_writer();
            let handle = tx.clone();
            let cycle = cycle.clone();
            rayon::spawn(move || {
                writer.write_instruction(&cycle, i);
                handle.send(1).unwrap();
            });
        }
        drop(tx);
        for msg in rx.iter() {
            assert!(msg == 1);
        }

        let trace = generator.trace_clone();

        for window in trace.windows_iter() {
            let mut window_parser = TraceWindowParser::new(window, &[], &[], &[]);
            assert_eq!(window_parser.local_slice().len(), L::num_columns());
            air.eval(&mut window_parser);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
