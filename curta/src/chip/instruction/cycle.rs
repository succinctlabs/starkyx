use serde::{Deserialize, Serialize};

use super::set::AirInstruction;
use super::Instruction;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cycle<F> {
    pub start_bit: BitRegister,
    pub end_bit: BitRegister,
    start_bit_witness: ElementRegister,
    end_bit_witness: ElementRegister,
    element: ElementRegister,
    group: Vec<F>,
}

pub struct Loop {
    num_iterations: usize,
    iterations_registers: ArrayRegister<BitRegister>,
}

impl Loop {
    pub fn get_iteration_reg(&self, index: usize) -> BitRegister {
        assert!(
            index < self.num_iterations,
            "trying to get an iteration register that is out of bounds"
        );
        self.iterations_registers.get(index)
    }
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn cycle(&mut self, length_log: usize) -> Cycle<L::Field> {
        let start_bit = self.alloc::<BitRegister>();
        let end_bit = self.alloc::<BitRegister>();
        let element = self.alloc::<ElementRegister>();
        let start_bit_witness = self.alloc::<ElementRegister>();
        let end_bit_witness = self.alloc::<ElementRegister>();
        let group = L::Field::two_adic_subgroup(length_log);
        let cycle = Cycle {
            start_bit,
            end_bit,
            element,
            start_bit_witness,
            end_bit_witness,
            group,
        };

        self.register_air_instruction_internal(AirInstruction::cycle(cycle.clone()));

        cycle
    }

    pub fn loop_instr(&mut self, num_iterations: usize) -> Loop {
        let iterations_registers = self.alloc_array::<BitRegister>(num_iterations);

        // Set the cycle 12 registers first row
        self.set_to_expression_first_row(
            &iterations_registers.get(0),
            ArithmeticExpression::from_constant(L::Field::from_canonical_usize(1)),
        );

        for i in 1..num_iterations {
            self.set_to_expression_first_row(
                &iterations_registers.get(i),
                ArithmeticExpression::from_constant(L::Field::from_canonical_usize(0)),
            );
        }

        // Set transition constraint for the cycle_12_registers
        for i in 0..num_iterations {
            let next_i = (i + 1) % num_iterations;

            self.set_to_expression_transition(
                &iterations_registers.get(next_i).next(),
                iterations_registers.get(i).expr(),
            );
        }

        Loop {
            num_iterations,
            iterations_registers,
        }
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

        // Impose compatibility of the bit and the group so that following relations hold:
        // start_bit = 1 <=> element == 1, end_bit = 1 <=> element == gen_inv

        // In order to achieve this, we impose the following constraints on the start_bit and the
        // end_bit:
        // 1. start_bit * (element - 1) = 0. This implies that start_bit = 1 => element = 1.
        // 2. end_bit * (element - generator_inv) = 0 <=> end_bit = 1 => element = generator_inv.
        // 3. The field value `element - (1 - start_bit)` is invertible. This implies that
        // 4. The field value `element - generator_inv * (1 - end_bit)` is invertible. This implies

        // Impose start_bit * (element - 1) = 0, this implies that start_bit = 1 => element = 1
        let start_bit = self.start_bit.eval(parser);
        let elem_minus_one = parser.sub_const(element, F::ONE);
        let start_bit_constraint = parser.mul(start_bit, elem_minus_one);
        parser.constraint(start_bit_constraint);

        // Impose `element - (1 - start_bit)` is invertible using the witness as an inverse.
        let start_bit_witness = self.start_bit_witness.eval(parser);
        let elem_minus_one_minus_start_bit = parser.add(elem_minus_one, start_bit);
        let mut start_bit_witness_constraint =
            parser.mul(elem_minus_one_minus_start_bit, start_bit_witness);
        start_bit_witness_constraint = parser.sub_const(start_bit_witness_constraint, F::ONE);
        parser.constraint(start_bit_witness_constraint);

        // First, we impose end_bit * (element - generator_inv)  = 0.
        let generator_inv = self.group.last().unwrap();
        let end_bit = self.end_bit.eval(parser);
        let elem_minus_gen_inv = parser.sub_const(element, *generator_inv);
        let end_bit_constraint = parser.mul(end_bit, elem_minus_gen_inv);
        parser.constraint(end_bit_constraint);

        // Second, we impose that the field value `element - generator_inv * (1 - end_bit)` is
        // invertible. This implies that element = generator_inv => end_bit = 1.
        let end_bit_witness = self.end_bit_witness.eval(parser);
        let end_bit_gen_inv = parser.mul_const(end_bit, *generator_inv);
        let elem_minus_gen_inv_minus_end_bit = parser.add(elem_minus_gen_inv, end_bit_gen_inv);
        let mut end_bit_witness_constraint =
            parser.mul(elem_minus_gen_inv_minus_end_bit, end_bit_witness);
        end_bit_witness_constraint = parser.sub_const(end_bit_witness_constraint, F::ONE);
        parser.constraint(end_bit_witness_constraint);
    }
}

impl<F: Field> Instruction<F> for Cycle<F> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let cycle = row_index % self.group.len();
        let element = self.group[cycle];
        let gen_inverse = *self.group.last().unwrap();
        writer.write(&self.element, &element, row_index);
        if cycle == 0 {
            writer.write(&self.start_bit, &F::ONE, row_index);
            writer.write(&self.end_bit, &F::ZERO, row_index);
            writer.write(&self.start_bit_witness, &element.inverse(), row_index);
            writer.write(
                &self.end_bit_witness,
                &(element - gen_inverse).inverse(),
                row_index,
            );
        } else if cycle == self.group.len() - 1 {
            writer.write(&self.start_bit, &F::ZERO, row_index);
            writer.write(&self.end_bit, &F::ONE, row_index);
            writer.write(
                &self.start_bit_witness,
                &(element - F::ONE).inverse(),
                row_index,
            );
            writer.write(&self.end_bit_witness, &element.inverse(), row_index);
        } else {
            writer.write(&self.start_bit, &F::ZERO, row_index);
            writer.write(&self.end_bit, &F::ZERO, row_index);
            writer.write(
                &self.start_bit_witness,
                &(element - F::ONE).inverse(),
                row_index,
            );
            writer.write(
                &self.end_bit_witness,
                &(element - gen_inverse).inverse(),
                row_index,
            );
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
    }

    #[test]
    fn test_cycle_instruction() {
        type L = CycleTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let cycle = builder.cycle(4);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 8;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let (tx, rx) = channel();
        for i in 0..num_rows {
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

        for window in trace.windows() {
            let mut window_parser = TraceWindowParser::new(window, &[], &[], &[]);
            assert_eq!(window_parser.local_slice().len(), L::num_columns());
            air.eval(&mut window_parser);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
