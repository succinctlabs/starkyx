//! Shift right instruction
//!
//!
//! a << (b + c) = (a << b) << c
//! a << (b + 2^i c) = (a << b) << 2^i c = ((a << b) << c) << 2^(i-1) c

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::Register;
use crate::chip::AirParameters;
pub use crate::math::prelude::*;

impl<L: AirParameters> AirBuilder<L> {
    pub fn shr(
        &mut self,
        a: &ArrayRegister<BitRegister>,
        b: &ArrayRegister<BitRegister>,
    ) -> ArrayRegister<BitRegister> {
        let result = self.alloc_array::<BitRegister>(a.len());
        self.set_shr(a, b, &result);
        result
    }

    pub fn set_shr(
        &mut self,
        a: &ArrayRegister<BitRegister>,
        b: &ArrayRegister<BitRegister>,
        result: &ArrayRegister<BitRegister>,
    ) {
        let n = a.len();
        let m = b.len();
        assert!(m <= n, "b must be shorter or eual length to a");

        let mut temp = *a;
        for (k, bit) in b.into_iter().enumerate() {
            // Calculate the shift (temp << 2^k)
            let num_shift_bits = 1 << k;

            let res = if k == m - 1 {
                *result
            } else {
                self.alloc_array::<BitRegister>(n)
            };

            // For i< NUM_BITS - num_shift_bits, we have shifted_res[i] = temp[i + num_shift_bits]
            for i in 0..(n - num_shift_bits) {
                self.set_select(
                    &bit,
                    &temp.get(i + num_shift_bits),
                    &temp.get(i),
                    &res.get(i),
                );
            }

            // For i >= NUM_BITS - num_shift_bits, we have shifted_res[i] = 0
            let one_minus_bit = ArithmeticExpression::one() - bit.expr();
            for i in (n - num_shift_bits)..n {
                let value = one_minus_bit.clone() * temp.get(i).expr();
                self.set_to_expression(&res.get(i), value);
            }
            temp = res;
        }
    }

    pub fn shl(
        &mut self,
        a: &ArrayRegister<BitRegister>,
        b: &ArrayRegister<BitRegister>,
    ) -> ArrayRegister<BitRegister> {
        let result = self.alloc_array::<BitRegister>(a.len());
        self.set_shl(a, b, &result);
        result
    }

    pub fn set_shl(
        &mut self,
        a: &ArrayRegister<BitRegister>,
        b: &ArrayRegister<BitRegister>,
        result: &ArrayRegister<BitRegister>,
    ) {
        let n = a.len();
        let m = b.len();
        assert!(m <= n, "b must be shorter or eual length to a");

        let mut temp = *a;
        for (k, bit) in b.into_iter().enumerate() {
            // Calculate the shift (temp << 2^k)
            let num_shift_bits = 1 << k;

            let res = if k == m - 1 {
                *result
            } else {
                self.alloc_array::<BitRegister>(n)
            };

            // For i< num_shift_bits, we have shifted_res[i] = zero
            let one_minus_bit = ArithmeticExpression::one() - bit.expr();
            for i in 0..num_shift_bits {
                let value = one_minus_bit.clone() * temp.get(i).expr();
                self.set_to_expression(&res.get(i), value);
            }

            // For i >= num_shift_bits, we have shifted_res[i] = temp[i - num_shift_bits]
            for i in num_shift_bits..n {
                self.set_select(
                    &bit,
                    &temp.get(i - num_shift_bits),
                    &temp.get(i),
                    &res.get(i),
                );
            }
            temp = res;
        }
    }
}

#[cfg(test)]
pub mod tests {

    use rand::{thread_rng, Rng};
    use serde::{Deserialize, Serialize};

    use super::*;
    pub use crate::chip::builder::tests::*;
    use crate::chip::uint::bytes::bit_operations::util::{bits_u8_to_val, u8_to_bits_le};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ShfitTest<const N: usize, const M: usize>;

    impl<const N: usize, const M: usize> AirParameters for ShfitTest<N, M> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = EmptyInstruction<GoldilocksField>;

        const NUM_FREE_COLUMNS: usize = 2 * N + M * N + N;
    }

    #[test]
    fn test_shr() {
        type F = GoldilocksField;
        type L = ShfitTest<N, M>;
        const M: usize = 3;
        const N: usize = 8;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let b = builder.alloc_array::<BitRegister>(M);
        let result = builder.shr(&a, &b);
        let expected = builder.alloc_array::<BitRegister>(N);

        builder.assert_expressions_equal(result.expr(), expected.expr());

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 9;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let mut rng = thread_rng();

        let to_bits_le = |x: u8| {
            let mut bits = [0u8; 8];
            for (i, bit) in bits.iter_mut().enumerate() {
                *bit = (x >> i) & 1;
            }
            bits
        };

        let to_val = |bits: &[u8]| bits.iter().enumerate().map(|(i, b)| b << i).sum::<u8>();
        for i in 0..num_rows {
            let a_val = rng.gen::<u8>();
            let b_val = rng.gen::<u8>() % 8;
            let a_bits = to_bits_le(a_val);
            let b_bits = to_bits_le(b_val);
            assert_eq!(a_val, to_val(&a_bits));
            let expected_val = a_val >> b_val;
            let expected_bits = to_bits_le(expected_val);
            writer.write_array(&a, a_bits.map(F::from_canonical_u8), i);
            writer.write_array(&b, b_bits.map(F::from_canonical_u8), i);
            writer.write_array(&expected, expected_bits.map(F::from_canonical_u8), i);
            writer.write_row_instructions(&generator.air_data, i);
        }

        let trace = generator.trace_clone();

        for window in trace.windows() {
            let mut window_parser = TraceWindowParser::new(window, &[], &[], &[]);
            air.eval(&mut window_parser);
        }

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }

    #[test]
    fn test_shl() {
        type F = GoldilocksField;
        type L = ShfitTest<N, M>;
        const M: usize = 3;
        const N: usize = 8;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let b = builder.alloc_array::<BitRegister>(M);
        let result = builder.shl(&a, &b);
        let expected = builder.alloc_array::<BitRegister>(N);

        builder.assert_expressions_equal(result.expr(), expected.expr());

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 9;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let mut rng = thread_rng();

        for i in 0..num_rows {
            let a_val = rng.gen::<u8>();
            let b_val = rng.gen::<u8>() % 8;
            let a_bits = u8_to_bits_le(a_val);
            let b_bits = u8_to_bits_le(b_val);
            assert_eq!(a_val, bits_u8_to_val(&a_bits));
            let expected_val = a_val << b_val;
            let expected_bits = u8_to_bits_le(expected_val);
            writer.write_array(&a, a_bits.map(F::from_canonical_u8), i);
            writer.write_array(&b, b_bits.map(F::from_canonical_u8), i);
            writer.write_array(&expected, expected_bits.map(F::from_canonical_u8), i);
            writer.write_row_instructions(&generator.air_data, i);
        }

        let trace = generator.trace_clone();

        for window in trace.windows() {
            let mut window_parser = TraceWindowParser::new(window, &[], &[], &[]);
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
