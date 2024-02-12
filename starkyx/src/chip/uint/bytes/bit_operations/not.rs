use serde::{Deserialize, Serialize};

use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
pub use crate::math::prelude::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Not<const NUM_BITS: usize> {
    pub a: ArrayRegister<BitRegister>,
    pub result: ArrayRegister<BitRegister>,
}

impl<AP: AirParser, const NUM_BITS: usize> AirConstraint<AP> for Not<NUM_BITS> {
    fn eval(&self, parser: &mut AP) {
        debug_assert_eq!(self.a.len(), NUM_BITS);
        debug_assert_eq!(self.result.len(), NUM_BITS);
        let a = self.a.eval_array::<_, NUM_BITS>(parser);
        let result = self.result.eval_array::<_, NUM_BITS>(parser);

        let one = parser.one();

        for (a, result) in a.into_iter().zip(result) {
            let one_minus_a = parser.sub(one, a);
            parser.assert_eq(one_minus_a, result);
        }
    }
}

impl<F: Field, const NUM_BITS: usize> Instruction<F> for Not<NUM_BITS> {
    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let a = writer.read_array::<_, NUM_BITS>(&self.a, row_index);
        let result = a.into_iter().map(|a| F::ONE - a).collect::<Vec<_>>();

        writer.write_array(&self.result, result, row_index);
    }

    fn write_to_air(&self, writer: &mut impl AirWriter<Field = F>) {
        let a = writer.read_array::<_, NUM_BITS>(&self.a);
        let result = a.into_iter().map(|a| F::ONE - a).collect::<Vec<_>>();

        writer.write_array(&self.result, result);
    }
}

#[cfg(test)]
pub mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NotTest<const N: usize>;

    impl<const N: usize> AirParameters for NotTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = Not<N>;

        const NUM_FREE_COLUMNS: usize = 3 * N;
    }

    #[test]
    fn test_bit_not() {
        type F = GoldilocksField;
        type L = NotTest<N>;
        type SC = PoseidonGoldilocksStarkConfig;
        const N: usize = 32;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let result = builder.alloc_array::<BitRegister>(N);
        let expected = builder.alloc_array::<BitRegister>(N);

        builder.assert_expressions_equal(result.expr(), expected.expr());

        let not = Not { a, result };
        builder.register_instruction(not);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 9;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_bits = [false; N].map(|_| rng.gen_bool(0.5));

            for (a, expected) in a_bits.iter().zip(expected) {
                writer.write(&expected, &F::from_canonical_u8(!a as u8), i);
            }

            writer.write_array(&a, a_bits.map(|b| F::from_canonical_u8(b as u8)), i);
            writer.write_instruction(&not, i);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
