//! A row accumulator
//!
//!
//!

pub mod constraint;
pub mod trace;

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::Register;
use crate::chip::AirParameters;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Accumulator<F, E> {
    pub(crate) challenges: ArrayRegister<CubicRegister>,
    values: Vec<ArithmeticExpression<F>>,
    digest: CubicRegister,
    _marker: PhantomData<(F, E)>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn accumulate<T: Register>(
        &mut self,
        challenges: &ArrayRegister<CubicRegister>,
        values: &[T],
    ) -> CubicRegister {
        let values = values.iter().map(|data| data.expr()).collect::<Vec<_>>();
        self.accumulate_expressions(challenges, &values)
    }

    pub fn set_accumulate_expressions(
        &mut self,
        challenges: &ArrayRegister<CubicRegister>,
        values: &[ArithmeticExpression<L::Field>],
        digest: &CubicRegister,
        global: bool,
    ) {
        let total_length = values.iter().map(|data| data.size).sum::<usize>();
        assert_eq!(
            total_length,
            challenges.len(),
            "Accumulator challenges and values must be the same size"
        );

        let accumulator = Accumulator {
            challenges: *challenges,
            values: values.to_vec(),
            digest: *digest,
            _marker: PhantomData,
        };

        self.accumulators.push(accumulator.clone());
        if global {
            self.global_constraints.push(accumulator.into());
        } else {
            self.constraints.push(accumulator.into());
        }
    }

    pub fn accumulate_expressions(
        &mut self,
        challenges: &ArrayRegister<CubicRegister>,
        values: &[ArithmeticExpression<L::Field>],
    ) -> CubicRegister {
        let total_length = values.iter().map(|data| data.size).sum::<usize>();
        assert_eq!(
            total_length,
            challenges.len(),
            "Accumulator challenges and values must be the same size"
        );

        let digest = self.alloc_extended::<CubicRegister>();
        self.set_accumulate_expressions(challenges, values, &digest, false);

        digest
    }

    pub fn accumulate_public_expressions(
        &mut self,
        challenges: &ArrayRegister<CubicRegister>,
        values: &[ArithmeticExpression<L::Field>],
    ) -> CubicRegister {
        let total_length = values.iter().map(|data| data.size).sum::<usize>();
        assert_eq!(
            total_length,
            challenges.len(),
            "Accumulator challenges and values must be the same size"
        );

        let digest = self.alloc_global::<CubicRegister>();
        self.set_accumulate_expressions(challenges, values, &digest, true);

        digest
    }
}

#[cfg(test)]
pub mod tests {
    use plonky2::field::types::Sample;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::register::element::ElementRegister;
    use crate::math::extension::cubic::element::CubicElement;
    use crate::math::prelude::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct AccumulatorTest;

    impl AirParameters for AccumulatorTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 6;

        type Instruction = EmptyInstruction<GoldilocksField>;
    }

    #[test]
    fn test_accumulation() {
        type L = AccumulatorTest;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_1 = builder.alloc::<ElementRegister>();
        let x_2 = builder.alloc::<ElementRegister>();

        let expr_0 = x_1.expr() + x_2.expr();
        let expr_1 = x_1.expr() * x_2.expr();
        let expr_2 = x_1.expr() - x_2.expr() + F::ONE;
        let expr_3 = ArithmeticExpression::from_constant(F::from_canonical_u32(42));

        let challenges = builder.alloc_array_challenge(2);

        let digest = builder.accumulate(&challenges, &[x_1, x_2]);

        let challenges_expr = builder.alloc_array_challenge(4);

        let digest_expr = builder.accumulate_expressions(
            &challenges_expr,
            &[
                expr_0.clone(),
                expr_1.clone(),
                expr_2.clone(),
                expr_3.clone(),
            ],
        );

        let zero = ArithmeticExpression::<F>::zero();

        let alphas = challenges.iter().map(|c| c.ext_expr());
        let alphas_expr = challenges_expr.iter().map(|c| c.ext_expr());

        let mut acc =
            CubicElement::<ArithmeticExpression<F>>([zero.clone(), zero.clone(), zero.clone()]);
        for (alpha, x) in alphas.zip([x_1, x_2].iter()) {
            let x_ext =
                CubicElement::<ArithmeticExpression<F>>([x.expr(), zero.clone(), zero.clone()]);
            acc = acc + alpha * x_ext;
        }

        let mut acc_expr =
            CubicElement::<ArithmeticExpression<F>>([zero.clone(), zero.clone(), zero.clone()]);
        for (alpha, x) in alphas_expr.zip([expr_0, expr_1, expr_2, expr_3].iter()) {
            let x_ext =
                CubicElement::<ArithmeticExpression<F>>([x.clone(), zero.clone(), zero.clone()]);
            acc_expr = acc_expr + alpha * x_ext;
        }

        for (a, b) in digest.as_base_array().iter().zip(acc.as_slice().iter()) {
            builder.assert_expressions_equal(a.expr(), b.clone());
        }

        for (a, b) in digest_expr
            .as_base_array()
            .iter()
            .zip(acc_expr.as_slice().iter())
        {
            builder.assert_expressions_equal(a.expr(), b.clone());
        }

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 10;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let writer = generator.new_writer();
        for i in 0..num_rows {
            writer.write(&x_1, &GoldilocksField::rand(), i);
            writer.write(&x_2, &GoldilocksField::rand(), i);
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }

    #[test]
    fn test_public_accumulation() {
        type L = AccumulatorTest;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_1 = builder.alloc_public::<ElementRegister>();
        let x_2 = builder.alloc_public::<ElementRegister>();
        let _ = builder.clock();

        let expr_0 = x_1.expr() + x_2.expr();
        let expr_1 = x_1.expr() * x_2.expr();
        let expr_2 = x_1.expr() - x_2.expr() + F::ONE;
        let expr_3 = ArithmeticExpression::from_constant(F::from_canonical_u32(42));

        let challenges_expr = builder.alloc_array_challenge(4);

        let digest_expr = builder.accumulate_public_expressions(
            &challenges_expr,
            &[
                expr_0.clone(),
                expr_1.clone(),
                expr_2.clone(),
                expr_3.clone(),
            ],
        );

        let zero = ArithmeticExpression::<F>::zero();
        let alphas_expr = challenges_expr.iter().map(|c| c.ext_expr());

        let mut acc_expr =
            CubicElement::<ArithmeticExpression<F>>([zero.clone(), zero.clone(), zero.clone()]);
        for (alpha, x) in alphas_expr.zip([expr_0, expr_1, expr_2, expr_3].iter()) {
            let x_ext =
                CubicElement::<ArithmeticExpression<F>>([x.clone(), zero.clone(), zero.clone()]);
            acc_expr = acc_expr + alpha * x_ext;
        }

        for (a, b) in digest_expr
            .as_base_array()
            .iter()
            .zip(acc_expr.as_slice().iter())
        {
            builder.assert_expressions_equal(a.expr(), b.clone());
        }

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 10;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let writer = generator.new_writer();
        for i in 0..num_rows {
            writer.write_row_instructions(&generator.air_data, i);
        }

        let public_inputs = vec![GoldilocksField::rand(); 2];

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
