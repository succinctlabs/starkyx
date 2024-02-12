use serde::{Deserialize, Serialize};

use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
pub use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct And<const NUM_BITS: usize> {
    pub a: ArrayRegister<BitRegister>,
    pub b: ArrayRegister<BitRegister>,
    pub result: ArrayRegister<BitRegister>,
}

impl<AP: AirParser, const NUM_BITS: usize> AirConstraint<AP> for And<NUM_BITS> {
    fn eval(&self, parser: &mut AP) {
        debug_assert_eq!(self.a.len(), NUM_BITS);
        debug_assert_eq!(self.b.len(), NUM_BITS);
        debug_assert_eq!(self.result.len(), NUM_BITS);
        let a = self.a.eval_array::<_, NUM_BITS>(parser);
        let b = self.b.eval_array::<_, NUM_BITS>(parser);
        let result = self.result.eval_array::<_, NUM_BITS>(parser);

        for ((a, b), result) in a.into_iter().zip(b).zip(result) {
            let ab = parser.mul(a, b);
            parser.assert_eq(ab, result);
        }
    }
}

impl<F: Field, const NUM_BITS: usize> Instruction<F> for And<NUM_BITS> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let a = writer.read_array::<_, NUM_BITS>(&self.a, row_index);
        let b = writer.read_array::<_, NUM_BITS>(&self.b, row_index);

        let result = a.into_iter().zip(b).map(|(a, b)| a * b);

        writer.write_array(&self.result, result, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let a = writer.read_array::<_, NUM_BITS>(&self.a);
        let b = writer.read_array::<_, NUM_BITS>(&self.b);

        let result = a.into_iter().zip(b).map(|(a, b)| a * b);

        writer.write_array(&self.result, result);
    }
}

#[cfg(test)]
pub mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::register::Register;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AndTest<const N: usize>;

    impl<const N: usize> AirParameters for AndTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = And<N>;

        const NUM_FREE_COLUMNS: usize = 4 * N + 1;
    }

    #[test]
    fn test_bit_and() {
        type F = GoldilocksField;
        type L = AndTest<N>;
        type SC = PoseidonGoldilocksStarkConfig;
        const N: usize = 8;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let b = builder.alloc_array::<BitRegister>(N);
        let result = builder.alloc_array::<BitRegister>(N);
        let expected = builder.alloc_array::<BitRegister>(N);

        builder.assert_expressions_equal(result.expr(), expected.expr());

        let and = And { a, b, result };
        builder.register_instruction(and);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_bits = [false; N].map(|_| rng.gen_bool(0.5));
            let b_bits = [false; N].map(|_| rng.gen_bool(0.5));

            for ((a, b), expected) in a_bits.iter().zip(b_bits.iter()).zip(expected) {
                let a_and_b = *a && *b;
                writer.write(&expected, &F::from_canonical_u8(a_and_b as u8), i);
            }

            writer.write_array(&a, a_bits.map(|b| F::from_canonical_u8(b as u8)), i);
            writer.write_array(&b, b_bits.map(|b| F::from_canonical_u8(b as u8)), i);
            writer.write_row_instructions(&generator.air_data, i);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }

    #[test]
    fn test_filtered_bit_and() {
        type F = GoldilocksField;
        type L = AndTest<N>;
        type SC = PoseidonGoldilocksStarkConfig;
        const N: usize = 8;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let b = builder.alloc_array::<BitRegister>(N);
        let result = builder.alloc_array::<BitRegister>(N);
        let filter = builder.alloc::<BitRegister>();
        let expected = builder.alloc_array::<BitRegister>(N);

        builder.assert_expression_zero((result.expr() - expected.expr()) * filter.expr());

        let and = And { a, b, result };
        builder.register_instruction_with_filter(and, filter.expr());

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_bits = [false; N].map(|_| rng.gen_bool(0.5));
            let b_bits = [false; N].map(|_| rng.gen_bool(0.5));
            let filter_val = rng.gen_bool(0.5);

            for ((a, b), expected) in a_bits.iter().zip(b_bits.iter()).zip(expected) {
                let a_and_b = *a && *b;
                writer.write(&expected, &F::from_canonical_u8(a_and_b as u8), i);
            }

            writer.write_array(&a, a_bits.map(|b| F::from_canonical_u8(b as u8)), i);
            writer.write_array(&b, b_bits.map(|b| F::from_canonical_u8(b as u8)), i);
            writer.write(&filter, &F::from_canonical_u8(filter_val as u8), i);
            writer.write_row_instructions(&generator.air_data, i);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
