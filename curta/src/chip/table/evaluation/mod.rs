use core::marker::PhantomData;

use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::constraint::Constraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::extension::ExtensionRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub mod constraint;
pub mod trace;

#[derive(Debug, Clone)]
pub struct Evaluation<F: Field, E: CubicParameters<F>> {
    pub beta: ExtensionRegister<3>,
    beta_powers: ExtensionRegister<3>,
    pub alphas: ArrayRegister<ExtensionRegister<3>>,
    pub values: Vec<ElementRegister>,
    pub filter: ArithmeticExpression<F>,
    accumulator: ExtensionRegister<3>,
    pub digest: ExtensionRegister<3>,
    _marker: PhantomData<(F, E)>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn evaluation<T: Register>(
        &mut self,
        values: &[T],
        filter: ArithmeticExpression<L::Field>,
        digest: ExtensionRegister<3>,
    ) -> Evaluation<L::Field, L::CubicParams> {
        // Get the running evaluation challenge
        let beta = self.alloc_challenge::<ExtensionRegister<3>>();
        let beta_powers = self.alloc_extended::<ExtensionRegister<3>>();
        // get the row accumulation challenge
        let alphas =
            self.alloc_challenge_array::<ExtensionRegister<3>>(values.len() * T::size_of());

        let mut elem_vals = vec![];
        for val in values {
            let elem_array =
                ArrayRegister::<ElementRegister>::from_register_unsafe(*val.register());
            for e in elem_array.into_iter() {
                elem_vals.push(e);
            }
        }

        if !self.is_extended(&digest) {
            panic!("Digest must be an extended register");
        }

        let accumulator = self.alloc_extended::<ExtensionRegister<3>>();

        let evaluation = Evaluation {
            beta,
            beta_powers,
            alphas,
            values: elem_vals,
            filter,
            accumulator,
            digest,
            _marker: PhantomData,
        };
        self.constraints
            .push(Constraint::evaluation(evaluation.clone()));
        self.evaluation_data.push(evaluation.clone());
        evaluation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip::builder::tests::*;

    #[derive(Debug, Clone)]
    pub struct EvalTest;

    impl const AirParameters for EvalTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;
        type Instruction = EmptyInstruction<GoldilocksField>;
        const NUM_ARITHMETIC_COLUMNS: usize = 2;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 20;

        fn num_rows_bits() -> usize {
            4
        }
    }

    #[test]
    fn test_evaluation() {
        type F = GoldilocksField;
        type L = EvalTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_0 = builder.alloc::<U16Register>();
        let x_1 = builder.alloc::<U16Register>();

        let cycle = builder.cycle(4);

        let acc = builder.alloc_extended::<ExtensionRegister<3>>();

        let _eval = builder.evaluation(&[x_0, x_1], ArithmeticExpression::one(), acc);

        let (air, _) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&[]);

        let (tx, rx) = channel();
        for i in 0..L::num_rows() {
            let writer = generator.new_writer();
            let handle = tx.clone();
            writer.write_instruction(&cycle, i);
            rayon::spawn(move || {
                writer.write(&x_0, &[F::ZERO], i);
                writer.write(&x_1, &[F::from_canonical_usize(i)], i);
                handle.send(1).unwrap();
            });
        }
        drop(tx);
        for msg in rx.iter() {
            assert!(msg == 1);
        }
        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
