use std::sync::Arc;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;

use crate::arithmetic::register::MemorySlice;
use crate::vars::StarkEvaluationVars;

#[derive(Clone, Debug)]
pub enum AffineExpression<F, const D: usize> {
    Input(MemorySlice),
    Const(F),
    Sum(Arc<AffineExpression<F, D>>, Arc<AffineExpression<F, D>>),
    ScalarMul(F, Arc<AffineExpression<F, D>>),
}

#[derive(Clone, Debug)]
pub enum QuadraticExpression<F, const D: usize> {
    Affine(AffineExpression<F, D>),
    Sum(
        Arc<QuadraticExpression<F, D>>,
        Arc<QuadraticExpression<F, D>>,
    ),
}

impl<F: RichField + Extendable<D>, const D: usize> AffineExpression<F, D> {
    pub fn input(input: MemorySlice) -> Self {
        AffineExpression::Input(input)
    }

    pub fn packed_generic<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: &StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<P>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            AffineExpression::Input(input) => input.packed_entries_slice(vars).to_vec(),
            AffineExpression::Const(constant) => {
                let s = FE::from_basefield(*constant);
                vec![P::from(s)]
            }
            AffineExpression::Sum(left, right) => {
                let left = left.packed_generic(vars);
                let right = right.packed_generic(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r + *l)
                    .collect()
            }

            AffineExpression::ScalarMul(scalar, expr) => {
                let expr = expr.packed_generic(vars);
                let s = FE::from_basefield(*scalar);
                expr.iter().map(|e| *e * s).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::{Field, Sample};
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::arithmetic::builder::ChipBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
    use crate::arithmetic::instruction::write::WriteInstruction;
    use crate::arithmetic::instruction::EqualityConstraint;
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug)]
    pub struct TestAffineExpression<F, const D: usize> {
        _marker: core::marker::PhantomData<F>,
    }

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D>
        for TestAffineExpression<F, D>
    {
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 3;

        type Instruction = WriteInstruction;
    }

    #[test]
    fn test_linear_exp_packed_generic() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = TestStark<TestAffineExpression<F, D>, F, D>;

        let mut builder = ChipBuilder::<TestAffineExpression<F, D>, F, D>::new();

        let input_1 = builder.get_local_memory(1).unwrap();
        let input_2 = builder.get_local_memory(1).unwrap();
        let output = builder.get_local_memory(1).unwrap();

        builder.write_raw_register(&input_1).unwrap();
        builder.write_raw_register(&input_2).unwrap();
        builder.write_raw_register(&output).unwrap();

        let expr = AffineExpression::Sum(
            Arc::new(AffineExpression::ScalarMul(
                F::ONE + F::ONE,
                Arc::new(AffineExpression::<F, D>::Input(input_1)),
            )),
            Arc::new(AffineExpression::<F, D>::Input(input_2)),
        );

        let equal_consr = EqualityConstraint::AffineConstraint(output, expr);

        builder.insert_raw_constraint(equal_consr);

        let (chip, spec) = builder.build();

        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        for i in 0..num_rows {
            let a_val = F::rand();
            let b_val = F::rand();
            let c_val = (F::ONE + F::ONE) * a_val + b_val;
            handle.write_unsafe_raw(i, &input_1, vec![a_val]).unwrap();
            handle.write_unsafe_raw(i, &input_2, vec![b_val]).unwrap();
            handle.write_unsafe_raw(i, &output, vec![c_val]).unwrap();
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows).unwrap();

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);

        // Verify proof as a stark
        let proof = prove::<F, C, S, D>(
            stark.clone(),
            &config,
            trace,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();
    }
}
