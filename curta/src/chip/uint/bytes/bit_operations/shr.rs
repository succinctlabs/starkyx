use core::iter::once;

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
pub struct Shr<const NUM_BITS: usize> {
    pub a: ArrayRegister<BitRegister>,
    pub carry: BitRegister,
    pub result: ArrayRegister<BitRegister>,
}

impl<AP: AirParser, const NUM_BITS: usize> AirConstraint<AP> for Shr<NUM_BITS> {
    fn eval(&self, parser: &mut AP) {
        debug_assert_eq!(self.a.len(), NUM_BITS);
        debug_assert_eq!(self.result.len(), NUM_BITS);
        let a = self.a.eval_array::<_, NUM_BITS>(parser);
        let result = self.result.eval_array::<_, NUM_BITS>(parser);

        let carry = a[0];
        let carry_val = self.carry.eval(parser);
        parser.assert_eq(carry_val, carry);
        for i in 0..NUM_BITS - 1 {
            parser.assert_eq(result[i], a[i + 1]);
        }
        parser.constraint(result[NUM_BITS - 1]);
    }
}

impl<F: Field, const NUM_BITS: usize> Instruction<F> for Shr<NUM_BITS> {
    fn inputs(&self) -> Vec<MemorySlice> {
        vec![*self.a.register(), *self.carry.register()]
    }

    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![*self.result.register()]
    }

    fn constraint_degree(&self) -> usize {
        1
    }

    fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
        let a = writer.read_array::<_, NUM_BITS>(&self.a, row_index);

        let result = a[1..].iter().copied().chain(once(F::ZERO));
        writer.write_array(&self.result, result, row_index);
        writer.write(&self.carry, &a[0], row_index);
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
    pub struct ShrTest<const N: usize>;

    impl<const N: usize> const AirParameters for ShrTest<N> {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = Shr<N>;

        const NUM_FREE_COLUMNS: usize = 4 * N;

        fn num_rows_bits() -> usize {
            9
        }
    }

    #[test]
    fn test_shr() {
        type F = GoldilocksField;
        type L = ShrTest<32>;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let a = builder.alloc_array::<BitRegister>(32);
        let carry = builder.alloc::<BitRegister>();
        let result = builder.alloc_array::<BitRegister>(32);
        let expected = builder.alloc_array::<BitRegister>(32);

        builder.assert_expressions_equal(result.expr(), expected.expr());

        let shr = Shr { a, carry, result };
        builder.register_instruction(shr);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        let mut rng = thread_rng();
        for i in 0..L::num_rows() {
            let a_bits = [false; 32].map(|_| rng.gen_bool(0.5));
            writer.write_array(&a, a_bits.map(|b| F::from_canonical_u8(b as u8)), i);
            let mut a_32 = 0;
            for (k, bit) in a_bits.iter().enumerate() {
                a_32 += (*bit as u32) << k;
            }

            let exp_u32 = a_32 >> 1;

            for j in 0..32 {
                let exp_bit = (exp_u32 >> j) & 1;
                writer.write(&expected.get(j), &F::from_canonical_u32(exp_bit), i);
            }
            assert!((exp_u32 >> 31) & 1 == 0);

            writer.write_instruction(&shr, i);
            let carry = writer.read(&carry, i);
            let carry_val = a_32 & 1;
            assert_eq!(carry, F::from_canonical_u32(carry_val));
        }

        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
