//! Benchmarks Starky proof generation for a range of columns, rows, and constraint degrees.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::permutation::PermutationPair;
use crate::stark::Stark;
use crate::util::trace_rows_to_poly_values;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
extern crate test;

#[derive(Copy, Clone)]
pub struct BenchmarkStark<
    F: RichField + Extendable<D>,
    const D: usize,
    const NUM_COLS: usize,
    const NUM_ROWS: usize,
    const CONSTRAINT_DEGREE: usize,
> {
    _phantom: PhantomData<F>,
}

impl<
        F: RichField + Extendable<D>,
        const D: usize,
        const NUM_COLS: usize,
        const NUM_ROWS: usize,
        const CONSTRAINT_DEGREE: usize,
    > BenchmarkStark<F, D, NUM_COLS, NUM_ROWS, CONSTRAINT_DEGREE>
{
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    // Generates a trace where the row transitions from [x_1, ..., x_n] -> [x'_1, ..., x'_n], where
    // n is the number of columns. The transition function is x'_i <- x_i ** degree. To test the
    // permutation argument, we set x_{n-1} and x_n in the first row to be 0 and 1. Then, we set
    // x_n in the last row to be 0.
    pub fn generate_trace(&self) -> Vec<PolynomialValues<F>> {
        // The first row will be [2, ..., 2, 0, 1].
        let mut first_row = [F::TWO; NUM_COLS];
        first_row[NUM_COLS - 2] = F::ZERO;
        first_row[NUM_COLS - 1] = F::ONE;
        let mut trace_rows = (0..NUM_ROWS)
            .scan(first_row, |acc, _| {
                let tmp = *acc;
                for i in 0..(NUM_COLS - 2) {
                    acc[i] = tmp[i].exp_u64(CONSTRAINT_DEGREE as u64)
                }
                acc[NUM_COLS - 2] = tmp[NUM_COLS - 2] + F::ONE;
                acc[NUM_COLS - 1] = tmp[NUM_COLS - 1] + F::ONE;
                Some(tmp)
            })
            .collect::<Vec<_>>();
        trace_rows[NUM_ROWS - 1][NUM_COLS - 1] = F::ZERO;
        trace_rows_to_poly_values(trace_rows)
    }
}

impl<
        F: RichField + Extendable<D>,
        const D: usize,
        const NUM_COLS: usize,
        const NUM_ROWS: usize,
        const CONSTRAINT_DEGREE: usize,
    > Stark<F, D> for BenchmarkStark<F, D, NUM_COLS, NUM_ROWS, CONSTRAINT_DEGREE>
{
    const COLUMNS: usize = NUM_COLS;
    const PUBLIC_INPUTS: usize = 1;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // Check public inputs.
        for i in 0..(NUM_COLS - 2) {
            yield_constr.constraint_first_row(vars.local_values[i] - vars.public_inputs[0]);
        }

        // xi' <- xi ** CONSTRAINT_DEGREE
        for i in 0..(NUM_COLS - 2) {
            let mut exp = P::ONES;
            for i in 0..CONSTRAINT_DEGREE {
                exp = exp * vars.local_values[i];
            }
            yield_constr.constraint_transition(vars.next_values[i] - exp)
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        // Check public inputs.
        for i in 0..(NUM_COLS - 2) {
            let constraint = builder.sub_extension(vars.local_values[i], vars.public_inputs[0]);
            yield_constr.constraint_first_row(builder, constraint);
        }

        for i in 0..(NUM_COLS - 2) {
            let mut exp = builder.one_extension();
            for i in 0..CONSTRAINT_DEGREE {
                exp = builder.mul_extension(exp, vars.local_values[i]);
            }
            let constraint = builder.sub_extension(vars.next_values[i], exp);
            yield_constr.constraint_transition(builder, constraint);
        }
    }

    fn constraint_degree(&self) -> usize {
        CONSTRAINT_DEGREE
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        vec![PermutationPair::singletons(NUM_COLS - 2, NUM_COLS - 1)]
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::extension::Extendable;
    use plonky2::field::types::Field;
    use plonky2::hash::hash_types::RichField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{
        AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig,
    };
    use plonky2::util::timing::TimingTree;

    use crate::benchmark::BenchmarkStark;
    use crate::config::StarkConfig;
    use crate::proof::StarkProofWithPublicInputs;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::stark::Stark;
    use crate::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};
    use crate::verifier::verify_stark_proof;

    #[test]
    fn test_benchmark_stark() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        const NUM_COLS: usize = 4;
        const NUM_ROWS: usize = 1 << 5;
        const CONSTRAINT_DEGREE: usize = 2;
        type S = BenchmarkStark<F, D, NUM_COLS, NUM_ROWS, CONSTRAINT_DEGREE>;

        let config = StarkConfig::standard_fast_config();
        let public_inputs = [F::TWO];
        let stark = S::new();
        let trace = stark.generate_trace();
        let proof = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            public_inputs,
            &mut TimingTree::default(),
        )?;

        verify_stark_proof(stark, proof, &config)
    }

    #[test]
    fn test_benchmark_stark_degree() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        const NUM_COLS: usize = 4;
        const NUM_ROWS: usize = 1 << 5;
        const CONSTRAINT_DEGREE: usize = 2;
        type S = BenchmarkStark<F, D, NUM_COLS, NUM_ROWS, CONSTRAINT_DEGREE>;

        let stark = S::new();
        test_stark_low_degree(stark)
    }

    #[test]
    fn test_benchmark_stark_circuit() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        const NUM_COLS: usize = 4;
        const NUM_ROWS: usize = 1 << 5;
        const CONSTRAINT_DEGREE: usize = 2;
        type S = BenchmarkStark<F, D, NUM_COLS, NUM_ROWS, CONSTRAINT_DEGREE>;

        let stark = S::new();
        test_stark_circuit_constraints::<F, C, S, D>(stark)
    }

    #[test]
    fn test_recursive_stark_verifier() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        const NUM_COLS: usize = 4;
        const NUM_ROWS: usize = 1 << 5;
        const CONSTRAINT_DEGREE: usize = 2;
        type S = BenchmarkStark<F, D, NUM_COLS, NUM_ROWS, CONSTRAINT_DEGREE>;

        let config = StarkConfig::standard_fast_config();
        let public_inputs = [F::TWO];
        let stark = S::new();
        let trace = stark.generate_trace();
        let proof = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            public_inputs,
            &mut TimingTree::default(),
        )?;
        verify_stark_proof(stark, proof.clone(), &config)?;

        recursive_proof::<F, C, S, C, D>(stark, proof, &config, true)
    }

    fn recursive_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        S: Stark<F, D> + Copy,
        InnerC: GenericConfig<D, F = F>,
        const D: usize,
    >(
        stark: S,
        inner_proof: StarkProofWithPublicInputs<F, InnerC, D>,
        inner_config: &StarkConfig,
        print_gate_counts: bool,
    ) -> Result<()>
    where
        InnerC::Hasher: AlgebraicHasher<F>,
        [(); S::COLUMNS]:,
        [(); S::PUBLIC_INPUTS]:,
        [(); C::Hasher::HASH_SIZE]:,
    {
        let circuit_config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(circuit_config);
        let mut pw = PartialWitness::new();
        let degree_bits = inner_proof.proof.recover_degree_bits(inner_config);
        let pt = add_virtual_stark_proof_with_pis(&mut builder, stark, inner_config, degree_bits);
        set_stark_proof_with_pis_target(&mut pw, &pt, &inner_proof);

        verify_stark_proof_circuit::<F, InnerC, S, D>(&mut builder, stark, pt, inner_config);

        if print_gate_counts {
            builder.print_gate_counts(0);
        }

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof)
    }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }
}
