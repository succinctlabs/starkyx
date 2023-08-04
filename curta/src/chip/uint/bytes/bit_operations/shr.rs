//! Shift right instruction
//!
//!
//! a << (b + c) = (a << b) << c
//! a << (b + 2^i c) = (a << b) << 2^i c = ((a << b) << c) << 2^(i-1) c

use itertools::Itertools;

use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::instruction::Instruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::RegisterSerializable;
use crate::chip::trace::writer::TraceWriter;
pub use crate::math::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct Shr<const NUM_BITS: usize, const LOG_N: usize> {
    pub a: ArrayRegister<BitRegister>,
    pub b: ArrayRegister<BitRegister>,
    pub results: [ArrayRegister<BitRegister>; LOG_N],
}

impl<AP: AirParser, const NUM_BITS: usize, const LOG_N: usize> AirConstraint<AP>
    for Shr<NUM_BITS, LOG_N>
{
    fn eval(&self, parser: &mut AP) {
        debug_assert_eq!(self.a.len(), NUM_BITS);
        debug_assert_eq!(self.b.len(), NUM_BITS);
        let a = self.a.eval_array::<_, NUM_BITS>(parser);
        let b = self.b.eval_array::<_, LOG_N>(parser);
        let results = self
            .results
            .iter()
            .map(|r| r.eval_array::<_, NUM_BITS>(parser))
            .collect::<Vec<_>>();

        let mut temp = a;
        let one = parser.one();
        for (k, (bit, result)) in b.into_iter().zip_eq(results).enumerate() {
            // Calculate the shift (intermediate value << 2^k)
            let num_shift_bits = 1 << k;
            let one_minus_bit = parser.sub(one, bit);

            // For i< NUM_BITS - num_shift_bits, we have shifted_res[i] = temp[i + num_shift_bits]
            for i in 0..(NUM_BITS - num_shift_bits) {
                let shifted_res = temp[i + num_shift_bits];
                let shifted_res_times_bit = parser.mul(bit, shifted_res);
                let temp_times_one_minus_bit = parser.mul(one_minus_bit, temp[i]);
                let expected_result = parser.add(shifted_res_times_bit, temp_times_one_minus_bit);
                parser.assert_eq(expected_result, result[i]);
            }

            // For i >= NUM_BITS - num_shift_bits, we have shifted_res[i] = 0
            for i in (NUM_BITS - num_shift_bits)..NUM_BITS {
                let temp_times_one_minus_bit = parser.mul(one_minus_bit, temp[i]);
                let expected_result = parser.add(bit, temp_times_one_minus_bit);
                parser.assert_eq(expected_result, result[i]);
            }

            temp = result;
        }
    }
}

impl<F: Field, const NUM_BITS: usize, const LOG_N: usize> Instruction<F> for Shr<NUM_BITS, LOG_N> {
    fn inputs(&self) -> Vec<MemorySlice> {
        vec![*self.a.register(), *self.b.register()]
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        self.results.map(|x| *x.register()).to_vec()
    }

    fn constraint_degree(&self) -> usize {
        2
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let a = writer.read_array::<_, NUM_BITS>(&self.a, row_index);
        let b = writer.read_array::<_, LOG_N>(&self.b, row_index);

        let mut temp = a;
        for (k, &bit) in b.iter().enumerate() {
            let num_shift_bits = 1 << k;
            let mut result = [F::ZERO; NUM_BITS];
            for i in 0..(NUM_BITS - num_shift_bits) {
                result[i] = bit * temp[i + num_shift_bits] + (F::ONE - bit) * temp[i];
            }
            for i in (NUM_BITS - num_shift_bits)..NUM_BITS {
                result[i] = bit + (F::ONE - bit) * temp[i];
            }
            writer.write_array(&self.results[k], &result, row_index);
            temp = result;
        }
    }
}

#[cfg(test)]
pub mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::builder::AirBuilder;
    use crate::chip::AirParameters;

    #[derive(Debug, Clone)]
    pub struct ShrTest<const N: usize, const LOG_N: usize>;

    impl<const N: usize, const LOG_N: usize> const AirParameters for ShrTest<N, LOG_N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = Shr<N, LOG_N>;

        const NUM_FREE_COLUMNS: usize = 4 * N;

        fn num_rows_bits() -> usize {
            9
        }
    }

    #[test]
    fn test_shr() {
        type F = GoldilocksField;
        type L = ShrTest<N, LOG_N>;
        const LOG_N: usize = 3;
        const N: usize = 8;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let b = builder.alloc_array::<BitRegister>(LOG_N);
        let results = [builder.alloc_array::<BitRegister>(N); LOG_N];
        let expected = builder.alloc_array::<BitRegister>(N);

        let shr = Shr { a, b, results };
        builder.register_instruction(shr);

        builder.assert_expressions_equal(results[LOG_N - 1].expr(), expected.expr());

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        let mut rng = thread_rng();

        let to_bits_le = |x: u8| {
            let mut bits = [0u8; 8];
            for i in 0..8 {
                bits[i] = (x >> i) & 1;
            }
            bits
        };

        let to_val = |bits: &[u8]| bits.iter().enumerate().map(|(i, b)| b << i).sum::<u8>();
        for i in 0..L::num_rows() {
            let a_val = rng.gen::<u8>();
            let a_bits = to_bits_le(a_val);
            let b_bits = [0, 0, 0];
            let b_val = to_val(&b_bits);
            assert_eq!(a_val, to_val(&a_bits));
            let expected_val = a_val >> b_val;
            let expected_bits = to_bits_le(expected_val);
            writer.write_array(&a, a_bits.map(|a| F::from_canonical_u8(a)), i);
            writer.write_array(&b, b_bits.map(|b| F::from_canonical_u8(b)), i);
            writer.write_array(&expected, expected_bits.map(|b| F::from_canonical_u8(b)), i);
            writer.write_instruction(&shr, i);
        }

        let trace = generator.trace_clone();

        for window in trace.windows_iter() {
            let mut window_parser = TraceWindowParser::new(window, &[], &[], &[]);
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
