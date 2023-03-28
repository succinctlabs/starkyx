//! Implements lookup tables using Halo2's lookup argument.
//! Reference: https://github.com/mir-protocol/plonky2/blob/main/evm/src/lookup.rs

use std::cmp::Ordering;

use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub(crate) fn eval_lookups<
    F: Field,
    P: PackedField<Scalar = F>,
    const COLS: usize,
    const PUBLIC_INPUTS: usize,
>(
    vars: StarkEvaluationVars<F, P, COLS, PUBLIC_INPUTS>,
    yield_constr: &mut ConstraintConsumer<P>,
    col_permuted_input: usize,
    col_permuted_table: usize,
) {
    let local_perm_input = vars.local_values[col_permuted_input];
    let next_perm_table = vars.next_values[col_permuted_table];
    let next_perm_input = vars.next_values[col_permuted_input];

    // A "vertical" diff between the local and next permuted inputs.
    let diff_input_prev = next_perm_input - local_perm_input;
    // A "horizontal" diff between the next permuted input and permuted table value.
    let diff_input_table = next_perm_input - next_perm_table;

    yield_constr.constraint(diff_input_prev * diff_input_table);

    // This is actually constraining the first row, as per the spec, since `diff_input_table`
    // is a diff of the next row's values. In the context of `constraint_last_row`, the next
    // row is the first row.
    yield_constr.constraint_last_row(diff_input_table);
}

pub(crate) fn eval_lookups_circuit<
    F: RichField + Extendable<D>,
    const D: usize,
    const COLS: usize,
    const PUBLIC_INPUTS: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    vars: StarkEvaluationTargets<D, COLS, PUBLIC_INPUTS>,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    col_permuted_input: usize,
    col_permuted_table: usize,
) {
    let local_perm_input = vars.local_values[col_permuted_input];
    let next_perm_table = vars.next_values[col_permuted_table];
    let next_perm_input = vars.next_values[col_permuted_input];

    // A "vertical" diff between the local and next permuted inputs.
    let diff_input_prev = builder.sub_extension(next_perm_input, local_perm_input);
    // A "horizontal" diff between the next permuted input and permuted table value.
    let diff_input_table = builder.sub_extension(next_perm_input, next_perm_table);

    let diff_product = builder.mul_extension(diff_input_prev, diff_input_table);
    yield_constr.constraint(builder, diff_product);

    // This is actually constraining the first row, as per the spec, since `diff_input_table`
    // is a diff of the next row's values. In the context of `constraint_last_row`, the next
    // row is the first row.
    yield_constr.constraint_last_row(builder, diff_input_table);
}

/// Given an input column and a table column, generate the permuted input and permuted table columns
/// used in the Halo2 permutation argument.
pub fn permuted_cols<F: PrimeField64>(inputs: &[F], table: &[F]) -> (Vec<F>, Vec<F>) {
    let n = inputs.len();

    // The permuted inputs do not have to be ordered, but we found that sorting was faster than
    // hash-based grouping. We also sort the table, as this helps us identify "unused" table
    // elements efficiently.

    // To compare elements, e.g. for sorting, we first need them in canonical form. It would be
    // wasteful to canonicalize in each comparison, as a single element may be involved in many
    // comparisons. So we will canonicalize once upfront, then use `to_noncanonical_u64` when
    // comparing elements.

    let sorted_inputs = inputs
        .iter()
        .map(|x| x.to_canonical())
        .sorted_unstable_by_key(|x| x.to_noncanonical_u64())
        .collect_vec();
    let sorted_table = table
        .iter()
        .map(|x| x.to_canonical())
        .sorted_unstable_by_key(|x| x.to_noncanonical_u64())
        .collect_vec();

    let mut unused_table_inds = Vec::with_capacity(n);
    let mut unused_table_vals = Vec::with_capacity(n);
    let mut permuted_table = vec![F::ZERO; n];
    let mut i = 0;
    let mut j = 0;
    while (j < n) && (i < n) {
        let input_val = sorted_inputs[i].to_noncanonical_u64();
        let table_val = sorted_table[j].to_noncanonical_u64();
        match input_val.cmp(&table_val) {
            Ordering::Greater => {
                unused_table_vals.push(sorted_table[j]);
                j += 1;
            }
            Ordering::Less => {
                if let Some(x) = unused_table_vals.pop() {
                    permuted_table[i] = x;
                } else {
                    unused_table_inds.push(i);
                }
                i += 1;
            }
            Ordering::Equal => {
                permuted_table[i] = sorted_table[j];
                i += 1;
                j += 1;
            }
        }
    }

    unused_table_vals.extend_from_slice(&sorted_table[j..n]);
    unused_table_inds.extend(i..n);

    for (ind, val) in unused_table_inds.into_iter().zip_eq(unused_table_vals) {
        permuted_table[ind] = val;
    }

    (sorted_inputs, permuted_table)
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use anyhow::Result;
    use plonky2::field::extension::{Extendable, FieldExtension};
    use plonky2::field::packed::PackedField;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::field::types::Field;
    use plonky2::hash::hash_types::RichField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{
        AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig,
    };
    use plonky2::util::timing::TimingTree;
    use plonky2::util::transpose;

    use super::{eval_lookups, eval_lookups_circuit};
    use crate::config::StarkConfig;
    use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
    use crate::lookup::permuted_cols;
    use crate::permutation::PermutationPair;
    use crate::proof::StarkProofWithPublicInputs;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::stark::Stark;
    use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
    use crate::verifier::verify_stark_proof;

    #[derive(Copy, Clone)]
    pub struct RangeCheckStark<F: RichField + Extendable<D>, const D: usize> {
        value: F,
        _phantom: PhantomData<F>,
    }

    impl<F: RichField + Extendable<D>, const D: usize> Default for RangeCheckStark<F, D> {
        fn default() -> Self {
            Self::new(F::ZERO)
        }
    }

    impl<F: RichField + Extendable<D>, const D: usize> RangeCheckStark<F, D> {
        pub fn new(value: F) -> Self {
            Self {
                value,
                _phantom: PhantomData,
            }
        }

        pub fn generate_trace(&self) -> Vec<PolynomialValues<F>> {
            const NUM_ROWS: usize = 1 << 4;

            // Set the trace table to all zeros.
            let mut trace_rows = (0..NUM_ROWS)
                .map(|_| vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO])
                .collect::<Vec<_>>();

            // Set the first column to be the range check counter. Also set the second column to be
            // the column that only holds self.value.
            for i in 0..trace_rows.len() {
                trace_rows[i][0] = F::from_canonical_usize(i);
                trace_rows[i][1] = self.value;
            }

            // Generate witnessed permuted column and table permutation.
            // Tranpose the table to be column-wise and then witness the permuted column and the
            // table column using Halo2's permutation argument.
            // Reference: https://hackmd.io/@sin7y/HyqKwhddj
            let mut trace_cols = transpose(&trace_rows);
            let (col_perm, table_perm) = permuted_cols(&trace_cols[1], &trace_cols[0]);
            trace_cols[2].copy_from_slice(&col_perm);
            trace_cols[3].copy_from_slice(&table_perm);
            trace_cols.into_iter().map(PolynomialValues::new).collect()
        }
    }

    impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RangeCheckStark<F, D> {
        const COLUMNS: usize = 4;
        const PUBLIC_INPUTS: usize = 0;

        fn eval_packed_generic<FE, P, const D2: usize>(
            &self,
            vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
            yield_constr: &mut ConstraintConsumer<P>,
        ) where
            FE: FieldExtension<D2, BaseField = F>,
            P: PackedField<Scalar = FE>,
        {
            eval_lookups(vars, yield_constr, 2, 3);
        }

        fn eval_ext_circuit(
            &self,
            builder: &mut CircuitBuilder<F, D>,
            vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
            yield_constr: &mut RecursiveConstraintConsumer<F, D>,
        ) {
            eval_lookups_circuit(builder, vars, yield_constr, 2, 3)
        }

        fn constraint_degree(&self) -> usize {
            2
        }

        fn permutation_pairs(&self) -> Vec<PermutationPair> {
            vec![
                PermutationPair::singletons(0, 3),
                PermutationPair::singletons(1, 2),
            ]
        }
    }

    #[test]
    fn test_range_check_stark() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = RangeCheckStark<F, D>;

        let config = StarkConfig::standard_fast_config();
        let public_inputs = [];
        let stark = S::new(F::ZERO);
        let trace = stark.generate_trace();
        let proof = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            public_inputs,
            &mut TimingTree::default(),
        )?;

        assert!(verify_stark_proof(stark, proof, &config).is_ok());
        Ok(())
    }

    #[test]
    fn test_range_check_stark_failure() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = RangeCheckStark<F, D>;

        let config = StarkConfig::standard_fast_config();
        let public_inputs = [];
        let stark = S::new(F::from_canonical_u64(64));
        let trace = stark.generate_trace();
        let proof = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            public_inputs,
            &mut TimingTree::default(),
        )?;

        assert!(verify_stark_proof(stark, proof, &config).is_err());
        Ok(())
    }

    #[test]
    fn test_recursive_stark_verifier() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = RangeCheckStark<F, D>;

        let config = StarkConfig::standard_fast_config();
        let public_inputs = [];
        let stark = S::new(F::ZERO);
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
}
