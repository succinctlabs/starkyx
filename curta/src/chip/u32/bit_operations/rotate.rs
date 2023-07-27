
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
pub use crate::math::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct RightRotate<const NUM_BITS: usize> {
    pub a: ArrayRegister<BitRegister>,
    pub b: ArrayRegister<BitRegister>,
    pub result: ArrayRegister<BitRegister>,
}

impl<AP: AirParser, const NUM_BITS: usize> AirConstraint<AP> for RightRotate<NUM_BITS> {
    fn eval(&self, parser: &mut AP) {
        debug_assert_eq!(self.a.len(), NUM_BITS);
        debug_assert_eq!(self.b.len(), NUM_BITS);
        debug_assert_eq!(self.result.len(), NUM_BITS);
        let _a = self.a.eval_array::<_, NUM_BITS>(parser);
        let _b = self.b.eval_array::<_, NUM_BITS>(parser);
        let _result = self.result.eval_array::<_, NUM_BITS>(parser);

        todo!()
    }
}

// impl<F: Field, const NUM_BITS: usize> Instruction<F> for And<NUM_BITS> {
//     fn inputs(&self) -> HashSet<MemorySlice> {
//         HashSet::from([*self.a.register(), *self.b.register()])
//     }

//     fn trace_layout(&self) -> Vec<MemorySlice> {
//         vec![*self.result.register()]
//     }

//     fn constraint_degree(&self) -> usize {
//         2
//     }

//     fn write(&self, writer: &TraceWriter<F>, row_index: usize) {
//         let a = writer.read_array::<_, NUM_BITS>(&self.a, row_index);
//         let b = writer.read_array::<_, NUM_BITS>(&self.b, row_index);

//         let result = a.into_iter().zip(b).map(|(a, b)| a * b);

//         writer.write_array(&self.result, result, row_index);
//     }
// }

// #[cfg(test)]
// pub mod tests {
//     use rand::{thread_rng, Rng};

//     use super::*;
//     pub use crate::chip::builder::tests::*;
//     use crate::chip::builder::AirBuilder;
//     use crate::chip::AirParameters;

//     #[derive(Debug, Clone)]
//     pub struct AndTest<const N: usize>;

//     impl<const N: usize> const AirParameters for AndTest<N> {
//         type Field = GoldilocksField;
//         type CubicParams = GoldilocksCubicParameters;

//         type Instruction = And<N>;

//         const NUM_FREE_COLUMNS: usize = 4 * N;

//         fn num_rows_bits() -> usize {
//             9
//         }
//     }

//     #[test]
//     fn test_bit_and() {
//         type F = GoldilocksField;
//         type L = AndTest<N>;
//         type SC = PoseidonGoldilocksStarkConfig;
//         const N: usize = 32;

//         let mut builder = AirBuilder::<L>::new();

//         let a = builder.alloc_array::<BitRegister>(N);
//         let b = builder.alloc_array::<BitRegister>(N);
//         let result = builder.alloc_array::<BitRegister>(N);
//         let expected = builder.alloc_array::<BitRegister>(N);

//         builder.assert_expressions_equal(result.expr(), expected.expr());

//         let and = And { a, b, result };
//         builder.register_instruction(and);

//         let air = builder.build();

//         let generator = ArithmeticGenerator::<L>::new(&[]);
//         let writer = generator.new_writer();

//         let mut rng = thread_rng();
//         for i in 0..L::num_rows() {
//             let a_bits = [false; N].map(|_| rng.gen_bool(0.5));
//             let b_bits = [false; N].map(|_| rng.gen_bool(0.5));

//             for ((a, b), expected) in a_bits.iter().zip(b_bits.iter()).zip(expected) {
//                 let a_and_b = *a && *b;
//                 writer.write(&expected, &F::from_canonical_u8(a_and_b as u8), i);
//             }

//             writer.write_array(&a, &a_bits.map(|b| F::from_canonical_u8(b as u8)), i);
//             writer.write_array(&b, &b_bits.map(|b| F::from_canonical_u8(b as u8)), i);
//             writer.write_instruction(&and, i);
//         }

//         let stark = Starky::<_, { L::num_columns() }>::new(air);
//         let config = SC::standard_fast_config(L::num_rows());

//         // Generate proof and verify as a stark
//         test_starky(&stark, &config, &generator, &[]);

//         // Test the recursive proof.
//         test_recursive_starky(stark, config, generator, &[]);
//     }
// }
