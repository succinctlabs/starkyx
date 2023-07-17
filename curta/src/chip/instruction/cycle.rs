use std::collections::HashSet;

use plonky2_maybe_rayon::IndexedParallelIterator;

use super::set::{AirInstruction, InstructionSet};
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

#[derive(Debug, Clone)]
pub struct Cycle<F> {
    pub bit: BitRegister,
    element: ElementRegister,
    group: Vec<F>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn cycle(&mut self, length_log: usize) -> Cycle<L::Field> {
        let bit = self.alloc::<BitRegister>();
        let element = self.alloc::<ElementRegister>();
        let group = L::Field::two_adic_subgroup(length_log);
        let cycle = Cycle {
            bit,
            element,
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
        // bit = 1 => element = 1 and otherwise bit = 0
        let bit = self.bit.eval(parser);
        let elem_minus_g = parser.sub_const(element, F::ONE);
        let bit_constraint = parser.mul(bit, elem_minus_g);
        parser.constraint(bit_constraint);
    }
}

impl<F: Field> Instruction<F> for Cycle<F> {
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![*self.bit.register(), *self.element.register()]
    }

    fn inputs(&self) -> HashSet<MemorySlice> {
        HashSet::new()
    }

    fn constraint_degree(&self) -> usize {
        2
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let cycle = row_index % self.group.len();
        writer.write_value(&self.element, &self.group[cycle], row_index);
        if cycle == 0 {
            writer.write_value(&self.bit, &F::ONE, row_index);
        } else {
            writer.write_value(&self.bit, &F::ZERO, row_index);
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use rand::thread_rng;

    use super::*;
    use crate::chip::builder::tests::*;

    #[derive(Clone, Debug)]
    pub struct CycleTest;

    impl const AirParameters for CycleTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = EmptyInstruction<GoldilocksField>;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 3;

        fn num_rows_bits() -> usize {
            8
        }
    }

    #[test]
    fn test_cycle_instruction() {
        type F = GoldilocksField;
        type L = CycleTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let cycle = builder.cycle(4);

        let (air, _) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&[]);
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
            let mut window_parser = TraceWindowParser::new(window, &[], &[]);
            assert_eq!(window_parser.local_slice().len(), L::num_columns());
            air.eval(&mut window_parser);
        }

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
