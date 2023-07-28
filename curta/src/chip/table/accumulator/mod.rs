//! A row accumulator
//!
//!
//!
//!

pub mod constraint;
pub mod trace;

use core::marker::PhantomData;

use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::RegisterSerializable;
use crate::chip::AirParameters;

#[derive(Debug, Clone)]
pub struct Accumulator<E> {
    pub(crate) challenges: ArrayRegister<CubicRegister>,
    values: Vec<MemorySlice>,
    digest: CubicRegister,
    _marker: PhantomData<E>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn accumulate<T: RegisterSerializable>(
        &mut self,
        challenges: &ArrayRegister<CubicRegister>,
        values: &[T],
    ) -> CubicRegister {
        let values = values
            .iter()
            .map(|data| *data.register())
            .collect::<Vec<_>>();
        let total_length = values.iter().map(|data| data.len()).sum::<usize>();
        assert_eq!(
            total_length,
            challenges.len(),
            "Accumulator challenges and values must be the same size"
        );

        let digest = self.alloc_extended::<CubicRegister>();

        let accumulator = Accumulator {
            challenges: *challenges,
            values,
            digest,
            _marker: PhantomData,
        };

        self.accumulators.push(accumulator.clone());
        self.constraints.push(accumulator.into());

        digest
    }
}

#[cfg(test)]
pub mod tests {
    use plonky2::field::types::Sample;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
    use crate::chip::register::Register;
    use crate::math::extension::cubic::element::CubicElement;

    #[derive(Debug, Clone)]
    struct AccumulatorTest;

    impl AirParameters for AccumulatorTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 3;

        type Instruction = EmptyInstruction<GoldilocksField>;

        fn num_rows_bits() -> usize {
            5
        }
    }

    #[test]
    fn test_accumulation() {
        type L = AccumulatorTest;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_1 = builder.alloc::<ElementRegister>();
        let x_2 = builder.alloc::<ElementRegister>();

        let challenges = builder.alloc_challenge_array(2);

        let digest = builder.accumulate(&challenges, &[x_1, x_2]);

        let zero = ArithmeticExpression::<F>::zero();

        let alphas = challenges.iter().map(|c| c.ext_expr());

        let mut acc =
            CubicElement::<ArithmeticExpression<F>>([zero.clone(), zero.clone(), zero.clone()]);
        for (alpha, x) in alphas.zip([x_1, x_2].iter()) {
            let x_ext =
                CubicElement::<ArithmeticExpression<F>>([x.expr(), zero.clone(), zero.clone()]);
            acc = acc + alpha * x_ext;
        }

        for (a, b) in digest.as_base_array().iter().zip(acc.as_slice().iter()) {
            builder.assert_expressions_equal(a.expr(), b.clone());
        }

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&[]);

        let writer = generator.new_writer();
        for i in 0..L::num_rows() {
            writer.write(&x_1, &GoldilocksField::rand(), i);
            writer.write(&x_2, &GoldilocksField::rand(), i);
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}