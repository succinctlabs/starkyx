//! Shift right instruction
//!
//!
//! a << (b + c) = (a << b) << c
//! a << (b + 2^i c) = (a << b) << 2^i c = ((a << b) << c) << 2^(i-1) c

use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::AirParameters;
pub use crate::math::prelude::*;

impl<L: AirParameters> AirBuilder<L> {
    pub fn rotate_right(
        &mut self,
        a: &ArrayRegister<BitRegister>,
        b: &ArrayRegister<BitRegister>,
    ) -> ArrayRegister<BitRegister> {
        let result = self.alloc_array::<BitRegister>(a.len());
        self.set_rotate_right(a, b, &result);
        result
    }

    pub fn set_rotate_right(
        &mut self,
        a: &ArrayRegister<BitRegister>,
        b: &ArrayRegister<BitRegister>,
        result: &ArrayRegister<BitRegister>,
    ) {
        let n = a.len();
        let m = b.len();
        assert!(m <= n, "b must be shorter or eual length to a");

        //    let n_log = n.ilog2() as usize;

        let mut temp = *a;
        for (k, bit) in b.into_iter().enumerate() {
            // Calculate temp.right_rotate(2^k) and set it to result if bit = 1
            let num_rotate_bits = 1 << k;

            let res = if k == m - 1 {
                *result
            } else {
                self.alloc_array::<BitRegister>(n)
            };

            for i in 0..n {
                self.set_select(
                    &bit,
                    &temp.get((i + num_rotate_bits) % n),
                    &temp.get(i),
                    &res.get(i),
                );
            }
            temp = res;
        }
    }

    pub fn rotate_left(
        &mut self,
        a: &ArrayRegister<BitRegister>,
        b: &ArrayRegister<BitRegister>,
    ) -> ArrayRegister<BitRegister> {
        let result = self.alloc_array::<BitRegister>(a.len());
        self.set_rotate_left(a, b, &result);
        result
    }

    pub fn set_rotate_left(
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
            // Calculate temp.right_rotate(2^k) and set it to result if bit = 1
            let num_shift_bits = (1 << k) % n;

            let res = if k == m - 1 {
                *result
            } else {
                self.alloc_array::<BitRegister>(n)
            };

            for i in 0..n {
                self.set_select(
                    &bit,
                    &temp.get((n + i - num_shift_bits) % n),
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

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RotateTest<const N: usize, const M: usize>;

    impl<const N: usize, const M: usize> AirParameters for RotateTest<N, M> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = EmptyInstruction<GoldilocksField>;

        const NUM_FREE_COLUMNS: usize = 2 * N + M * N + N;
    }

    #[test]
    fn test_rotate_right() {
        type F = GoldilocksField;
        type L = RotateTest<N, LOG_N>;
        const LOG_N: usize = 8;
        const N: usize = 8;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let b = builder.alloc_array::<BitRegister>(LOG_N);
        let result = builder.alloc_array::<BitRegister>(N);
        builder.set_rotate_right(&a, &b, &result);
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
            let b_val = rng.gen::<u8>();
            let a_bits = to_bits_le(a_val);
            let b_bits = to_bits_le(b_val);
            assert_eq!(a_val, to_val(&a_bits));
            let expected_val = a_val.rotate_right(b_val as u32);
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
    fn test_rotate_left() {
        type F = GoldilocksField;
        type L = RotateTest<N, M>;
        const M: usize = 8;
        const N: usize = 8;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(N);
        let b = builder.alloc_array::<BitRegister>(M);
        let result = builder.rotate_left(&a, &b);
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
            let b_val = rng.gen::<u8>();
            let a_bits = to_bits_le(a_val);
            let b_bits = to_bits_le(b_val);
            assert_eq!(a_val, to_val(&a_bits));
            assert_eq!(b_val, to_val(&b_bits));
            let expected_val = a_val.rotate_left(b_val as u32);
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
}
