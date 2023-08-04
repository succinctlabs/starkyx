use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
pub use crate::math::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct Adc<const NUM_BITS: usize> {
    pub a: ArrayRegister<BitRegister>,
    pub b: ArrayRegister<BitRegister>,
    carry: Option<BitRegister>,
    pub result: ArrayRegister<BitRegister>,
    result_carry: Option<BitRegister>,
}

impl<AP: AirParser, const NUM_BITS: usize> AirConstraint<AP> for Adc<NUM_BITS> {
    fn eval(&self, parser: &mut AP) {
        debug_assert_eq!(self.a.len(), NUM_BITS);
        debug_assert_eq!(self.b.len(), NUM_BITS);
        debug_assert_eq!(self.result.len(), NUM_BITS);
        let a = self.a.eval_array::<_, NUM_BITS>(parser);
        let b = self.b.eval_array::<_, NUM_BITS>(parser);
        let carry = match self.carry {
            Some(bit) => bit.eval(parser),
            None => parser.zero(),
        };

        let result = self.result.eval_array::<_, NUM_BITS>(parser);

        let mut carry = carry;
        for ((a, b), result) in a.into_iter().zip(b).zip(result) {
            let mut a_plus_b_plus_carry = parser.add(a, b);
            a_plus_b_plus_carry = parser.add(a_plus_b_plus_carry, carry);

            let ab = parser.mul(a, b);
            let ac = parser.mul(a, carry);
            let bc = parser.mul(b, carry);

        }
        if let Some(result_carry) = self.result_carry {
            let res_carry = result_carry.eval(parser);
            parser.assert_eq(res_carry, carry);
        }
    }
}

impl<F: Field, const NUM_BITS: usize> Instruction<F> for Adc<NUM_BITS> {
    fn inputs(&self) -> Vec<MemorySlice> {
        vec![*self.a.register(), *self.b.register()]
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![*self.result.register()]
    }

    fn constraint_degree(&self) -> usize {
        2
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let a = writer.read_array::<_, NUM_BITS>(&self.a, row_index);
        let b = writer.read_array::<_, NUM_BITS>(&self.b, row_index);

        // let result = a.into_iter().zip(b).map(|(a, b)| a * b);

        // writer.write_array(&self.result, result, row_index);
    }
}

#[cfg(test)]
pub mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::uint::bytes::bit_operations::test_helpers::u8_to_bits_le;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone)]
    pub struct ADCTest<const N: usize>;

    impl<const N: usize> const AirParameters for ADCTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = Adc<N>;

        const NUM_FREE_COLUMNS: usize = 4 * N;

        fn num_rows_bits() -> usize {
            9
        }
    }

    #[test]
    fn test_bit_adc() {
        type F = GoldilocksField;
        type L = ADCTest<N>;
        type SC = PoseidonGoldilocksStarkConfig;
        const N: usize = 8;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let b = builder.alloc_array::<BitRegister>(N);
        let result = builder.alloc_array::<BitRegister>(N);
        let expected = builder.alloc_array::<BitRegister>(N);

        builder.assert_expressions_equal(result.expr(), expected.expr());

        let adc = Adc {
            a,
            b,
            carry: None,
            result,
            result_carry: None,
        };
        builder.register_instruction(adc);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        let mut rng = thread_rng();
        for i in 0..L::num_rows() {
            let a_val = rng.gen::<u8>();
            let b_val = rng.gen::<u8>();
            let a_bits = u8_to_bits_le(a_val);
            let b_bits = u8_to_bits_le(b_val);

            let (expected_val, _) = a_val.carrying_add(b_val, false);
            let expected_bits = u8_to_bits_le(expected_val);

            writer.write_array(&a, a_bits.map(|b| F::from_canonical_u8(b as u8)), i);
            writer.write_array(&b, b_bits.map(|b| F::from_canonical_u8(b as u8)), i);
            writer.write_array(
                &result,
                expected_bits.map(|b| F::from_canonical_u8(b as u8)),
                i,
            );
            writer.write_array(
                &expected,
                expected_bits.map(|b| F::from_canonical_u8(b as u8)),
                i,
            );
            writer.write_row_instructions(&air, i);
        }

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
