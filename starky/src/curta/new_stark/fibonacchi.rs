use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

use super::Stark;
use crate::curta::trace::types::ConstantGenerator;

#[derive(Debug, Clone)]
pub struct FibonacciStark<F, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> FibonacciStark<F, D> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

    pub fn generator(&self, x0: F, x1: F, num_rows: usize) -> ConstantGenerator<F> {
        let mut trace_rows = (0..num_rows)
            .scan([x0, x1, F::ZERO, F::ONE], |acc, _| {
                let tmp = *acc;
                acc[0] = tmp[1];
                acc[1] = tmp[0] + tmp[1];
                acc[2] = tmp[2] + F::ONE;
                acc[3] = tmp[3] + F::ONE;
                Some(tmp)
            })
            .map(|arr| arr.to_vec())
            .collect::<Vec<_>>();
        trace_rows[num_rows - 1][3] = F::ZERO; // So that column 2 and 3 are permutation of one another.
        ConstantGenerator::from_rows(trace_rows)
    }
}

pub fn fibonacci<F: Field>(n: usize, x0: F, x1: F) -> F {
    (0..n).fold((x0, x1), |x, _| (x.1, x.0 + x.1)).1
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D, 1> for FibonacciStark<F, D> {
    const COLUMNS: usize = 4;
    const PUBLIC_INPUTS: usize = 3;
    const CHALLENGES: usize = 0;

    /// Columns for each round
    fn round_data(&self) -> [(usize, usize); 1] {
        [(0, 4)]
    }

    /// The number of challenges per round
    fn num_challenges(&self, _round: usize) -> usize {
        0
    }

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: super::vars::StarkEvaluationVars<
            FE,
            P,
            { Self::COLUMNS },
            { Self::PUBLIC_INPUTS },
            { Self::CHALLENGES },
        >,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: plonky2::field::extension::FieldExtension<D2, BaseField = F>,
        P: plonky2::field::packed::PackedField<Scalar = FE>,
    {
        // Check public inputs.
        yield_constr.constraint_first_row(vars.local_values[0] - vars.public_inputs[0]);
        yield_constr.constraint_first_row(vars.local_values[1] - vars.public_inputs[1]);
        yield_constr.constraint_last_row(vars.local_values[1] - vars.public_inputs[2]);

        // x0' <- x1
        yield_constr.constraint_transition(vars.next_values[0] - vars.local_values[1]);
        // x1' <- x0 + x1
        yield_constr.constraint_transition(
            vars.next_values[1] - vars.local_values[0] - vars.local_values[1],
        );
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        vars: super::vars::StarkEvaluationTargets<
            D,
            { Self::COLUMNS },
            { Self::PUBLIC_INPUTS },
            { Self::CHALLENGES },
        >,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // Check public inputs.
        let pis_constraints = [
            builder.sub_extension(vars.local_values[0], vars.public_inputs[0]),
            builder.sub_extension(vars.local_values[1], vars.public_inputs[1]),
            builder.sub_extension(vars.local_values[1], vars.public_inputs[2]),
        ];
        yield_constr.constraint_first_row(builder, pis_constraints[0]);
        yield_constr.constraint_first_row(builder, pis_constraints[1]);
        yield_constr.constraint_last_row(builder, pis_constraints[2]);

        // x0' <- x1
        let first_col_constraint = builder.sub_extension(vars.next_values[0], vars.local_values[1]);
        yield_constr.constraint_transition(builder, first_col_constraint);
        // x1' <- x0 + x1
        let second_col_constraint = {
            let tmp = builder.sub_extension(vars.next_values[1], vars.local_values[0]);
            builder.sub_extension(tmp, vars.local_values[1])
        };
        yield_constr.constraint_transition(builder, second_col_constraint);
    }

    fn constraint_degree(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::new_stark::prover::prove;
    use crate::curta::new_stark::verifier::verify_stark_proof;
    use crate::curta::trace::types::ConstantGenerator;

    #[test]
    fn test_new_fibonacci_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = FibonacciStark<F, D>;
        type T = ConstantGenerator<F>;

        let config = StarkConfig::standard_fast_config();
        let num_rows = 1 << 5;
        let public_inputs = [F::ZERO, F::ONE, fibonacci(num_rows - 1, F::ZERO, F::ONE)];
        let stark = S::new();

        let trace_generator = stark.generator(public_inputs[0], public_inputs[1], num_rows);
        let proof = prove::<F, C, S, T, D, 1>(
            stark.clone(),
            &config,
            trace_generator,
            num_rows,
            public_inputs,
            &mut TimingTree::default(),
        )
        .unwrap();

        verify_stark_proof(stark, proof, &config).unwrap();
    }
}
