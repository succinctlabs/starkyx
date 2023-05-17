//! Range checks based on logarithmic derivatives.
//!

use alloc::collections::VecDeque;
use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::builder::StarkBuilder;
use super::chip::StarkParameters;
use super::constraint::arithmetic::ArithmeticExpression;
use super::extension::cubic::array::CubicArray;
use super::extension::cubic::gadget::CubicGadget;
use super::extension::cubic::register::CubicElementRegister;
use super::extension::cubic::{CubicExtension, CubicParameters};
use super::register::{ArrayRegister, ElementRegister, Register, RegisterSerializable};
use super::trace::TraceGenerator;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

const BETAS: [u64; 3] = [
    17800306513594245228,
    422882772345461752,
    14491510587541603695,
];

#[derive(Debug, Clone)]
pub struct RangeCheckData {
    table: ElementRegister,
    multiplicity: ElementRegister,
    multiplicity_table_log: CubicElementRegister,
    row_accumulators: ArrayRegister<CubicElementRegister>,
    log_lookup_accumulator: CubicElementRegister,
}

impl RangeCheckData {
    pub fn packed_generic_constraints<
        L: StarkParameters<F, D>,
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let beta = CubicArray([
            P::from(FE::from_canonical_u64(BETAS[0])),
            P::from(FE::from_canonical_u64(BETAS[1])),
            P::from(FE::from_canonical_u64(BETAS[2])),
        ]);

        let multiplicity = CubicArray::from_base(
            self.multiplicity.register().packed_generic_vars(vars)[0],
            P::ZEROS,
        );
        let table =
            CubicArray::from_base(self.table.register().packed_generic_vars(vars)[0], P::ZEROS);
        let multiplicity_table_log = CubicArray::from_slice(
            self.multiplicity_table_log
                .register()
                .packed_generic_vars(vars),
        );

        let mult_table_constraint = multiplicity - multiplicity_table_log * (beta - table);
        for consr in mult_table_constraint.0 {
            yield_constr.constraint(consr);
        }

        let mut row_acc_iter = self
            .row_accumulators
            .iter()
            .map(|r| CubicArray::from_slice(r.register().packed_generic_vars(vars)));

        let range_idx = L::NUM_FREE_COLUMNS..(L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS);
        let mut range_pairs = range_idx
            .step_by(2)
            .map(|k| (vars.local_values[k], vars.local_values[k + 1]))
            .map(|(a, b)| {
                (
                    CubicArray::from_base(a, P::ZEROS),
                    CubicArray::from_base(b, P::ZEROS),
                )
            })
            .map(|(a, b)| (beta - a, beta - b));

        let ((beta_minus_a_0, beta_minus_b_0), acc_0) =
            (range_pairs.next().unwrap(), row_acc_iter.next().unwrap());

        let constr_0 = beta_minus_a_0 + beta_minus_b_0 - acc_0 * beta_minus_a_0 * beta_minus_b_0;
        for consr in constr_0.0 {
            yield_constr.constraint(consr);
        }

        let mut prev = acc_0;
        for ((beta_minus_a, beta_minus_b), acc) in range_pairs.zip(row_acc_iter) {
            let constraint =
                (beta_minus_a + beta_minus_b) - (acc - prev) * beta_minus_a * beta_minus_b;
            for consr in constraint.0 {
                yield_constr.constraint(consr);
            }
            prev = acc;
        }

        let log_lookup_accumulator = CubicArray::from_slice(
            self.log_lookup_accumulator
                .register()
                .packed_generic_vars(vars),
        );
        let log_lookup_accumulator_next = CubicArray::from_slice(
            self.log_lookup_accumulator
                .next()
                .register()
                .packed_generic_vars(vars),
        );

        let acc_transition_constraint =
            log_lookup_accumulator_next - log_lookup_accumulator - prev + multiplicity_table_log;
        for consr in acc_transition_constraint.0 {
            yield_constr.constraint_transition(consr);
        }

        let acc_first_row_constraint = log_lookup_accumulator;
        for consr in acc_first_row_constraint.0 {
            yield_constr.constraint_first_row(consr);
        }

        let acc_last_row_constraint = log_lookup_accumulator + prev - multiplicity_table_log;
        for consr in acc_last_row_constraint.0 {
            yield_constr.constraint_last_row(consr);
        }
    }

    pub fn ext_circuit_constraints<
        L: StarkParameters<F, D>,
        F: RichField + Extendable<D>,
        const D: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let beta = CubicGadget::const_extension(
            builder,
            [
                F::Extension::from(F::from_canonical_u64(BETAS[0])),
                F::Extension::from(F::from_canonical_u64(BETAS[1])),
                F::Extension::from(F::from_canonical_u64(BETAS[2])),
            ],
        );

        let multiplicity = CubicGadget::from_base_extension(
            builder,
            self.multiplicity.register().ext_circuit_vars(vars)[0],
        );

        let table = CubicGadget::from_base_extension(
            builder,
            self.table.register().ext_circuit_vars(vars)[0],
        );
        let multiplicity_table_log = CubicArray::from_slice(
            self.multiplicity_table_log
                .register()
                .ext_circuit_vars(vars),
        );

        let beta_minus_table = CubicGadget::sub_extension(builder, &beta, &table);
        let mut mult_table_constraint =
            CubicGadget::mul_extension(builder, &multiplicity_table_log, &beta_minus_table);
        mult_table_constraint =
            CubicGadget::sub_extension(builder, &multiplicity, &mult_table_constraint);
        for consr in mult_table_constraint.0 {
            yield_constr.constraint(builder, consr);
        }

        let mut row_acc_vec = self
            .row_accumulators
            .iter()
            .map(|r| CubicArray::from_slice(r.register().ext_circuit_vars(vars)))
            .collect::<VecDeque<_>>();

        let range_idx = L::NUM_FREE_COLUMNS..(L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS);
        let mut range_pairs = range_idx
            .step_by(2)
            .map(|k| (vars.local_values[k], vars.local_values[k + 1]))
            .map(|(a, b)| {
                (
                    CubicGadget::from_base_extension(builder, a),
                    CubicGadget::from_base_extension(builder, b),
                )
            })
            .collect::<Vec<_>>()
            .iter()
            .map(|(a, b)| {
                (
                    CubicGadget::sub_extension(builder, &beta, a),
                    CubicGadget::sub_extension(builder, &beta, b),
                )
            })
            .collect::<VecDeque<_>>();

        let ((beta_minus_a_0, beta_minus_b_0), acc_0) =
            (range_pairs.pop_front().unwrap(), row_acc_vec.pop_front().unwrap());

        let beta_minus_a_b = CubicGadget::mul_extension(builder, &beta_minus_a_0, &beta_minus_b_0);
        let acc_beta_m_ab = CubicGadget::mul_extension(builder, &acc_0, &beta_minus_a_b);
        let mut constr_0 = CubicGadget::add_extension(builder, &beta_minus_a_0, &beta_minus_b_0);
        constr_0 = CubicGadget::sub_extension(builder, &constr_0, &acc_beta_m_ab);
        for consr in constr_0.0 {
            yield_constr.constraint(builder, consr);
        }

        let mut prev = acc_0;
        for ((beta_minus_a, beta_minus_b), acc) in range_pairs.iter().zip(row_acc_vec.iter()) {
            let acc_minus_prev = CubicGadget::sub_extension(builder, acc, &prev);
            let mut product = CubicGadget::mul_extension(builder, beta_minus_a, &beta_minus_b);
            product = CubicGadget::mul_extension(builder, &product, &acc_minus_prev);
            let mut constraint = CubicGadget::add_extension(builder, &beta_minus_a, &beta_minus_b);
            constraint = CubicGadget::sub_extension(builder, &constraint, &product);
            for consr in constraint.0 {
                yield_constr.constraint(builder, consr);
            }
            prev = *acc;
        }

        let log_lookup_accumulator = CubicArray::from_slice(
            self.log_lookup_accumulator
                .register()
                .ext_circuit_vars(vars),
        );
        let log_lookup_accumulator_next = CubicArray::from_slice(
            self.log_lookup_accumulator
                .next()
                .register()
                .ext_circuit_vars(vars),
        );

        let mut acc_transition_constraint = CubicGadget::sub_extension(
            builder,
            &log_lookup_accumulator_next,
            &log_lookup_accumulator,
        );
        acc_transition_constraint =
            CubicGadget::sub_extension(builder, &acc_transition_constraint, &prev);
        acc_transition_constraint = CubicGadget::add_extension(
            builder,
            &acc_transition_constraint,
            &multiplicity_table_log,
        );
        for consr in acc_transition_constraint.0 {
            yield_constr.constraint_transition(builder, consr);
        }

        let acc_first_row_constraint = log_lookup_accumulator;
        for consr in acc_first_row_constraint.0 {
            yield_constr.constraint_first_row(builder, consr);
        }

        let mut acc_last_row_constraint =
            CubicGadget::add_extension(builder, &log_lookup_accumulator, &prev);
        acc_last_row_constraint =
            CubicGadget::sub_extension(builder, &acc_last_row_constraint, &multiplicity_table_log);
        for consr in acc_last_row_constraint.0 {
            yield_constr.constraint_last_row(builder, consr);
        }
    }
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub(crate) fn arithmetic_range_checks(&mut self) {
        // assign registers for table, multiplicity and accumulators
        let table = self.alloc::<ElementRegister>();
        let multiplicity = self.alloc::<ElementRegister>();
        let multiplicity_table_log = self.alloc::<CubicElementRegister>();
        let log_lookup_accumulator = self.alloc::<CubicElementRegister>();

        let one = || -> ArithmeticExpression<F, D> { ArithmeticExpression::one() };
        let zero = || -> ArithmeticExpression<F, D> { ArithmeticExpression::zero() };

        // Table constraints
        self.assert_expressions_equal_first_row(table.expr(), zero());
        self.assert_expressions_equal_transition(table.expr() + one(), table.next().expr());

        assert_eq!(
            L::NUM_ARITHMETIC_COLUMNS % 2,
            0,
            "The number of arithmetic columns must be even"
        );
        let num_accumulators = L::NUM_ARITHMETIC_COLUMNS / 2;
        let row_accumulators = self.alloc_array::<CubicElementRegister>(num_accumulators);


        self.range_data = Some(RangeCheckData {
            table,
            multiplicity,
            multiplicity_table_log,
            row_accumulators,
            log_lookup_accumulator,
        });
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceGenerator<F, D> {
    pub fn write_arithmetic_range_checks<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        trace_rows: &mut Vec<Vec<F>>,
        range_data: &RangeCheckData,
        range_idx: (usize, usize),
    ) -> Result<()> {
        // write table constraints
        let table = range_data.table.register();
        for i in 0..num_rows {
            let value = F::from_canonical_usize(i);
            table.assign(trace_rows, 0, &mut vec![value], i);
        }

        // Get the challenge
        let betas = [
            F::from_canonical_u64(BETAS[0]),
            F::from_canonical_u64(BETAS[1]),
            F::from_canonical_u64(BETAS[2]),
        ];
        let beta = CubicExtension::<F, E>::from(betas);

        // Calculate multiplicities
        let mut multiplicities = vec![F::ZERO; num_rows];

        for row in trace_rows.iter() {
            for value in row[range_idx.0..range_idx.1].iter() {
                // sum_check += one_ext / (beta - (*value).into());
                let value = value.to_canonical_u64() as usize;
                assert!(value < 1 << 16);
                multiplicities[value] += F::ONE;
            }
        }

        // Write multiplicities into the trace
        let multiplicity = range_data.multiplicity.register();
        for i in 0..num_rows {
            multiplicity.assign(trace_rows, 0, &mut vec![multiplicities[i]], i);
        }

        // Write multiplicity inverse constraint
        let mult_table_log_entries = multiplicities
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let table = CubicExtension::from(F::from_canonical_usize(i));
                CubicExtension::from(*x) / (beta - table)
            })
            .collect::<Vec<_>>();

        let mult_table_log = range_data.multiplicity_table_log.register();
        for (i, value) in mult_table_log_entries.iter().enumerate() {
            mult_table_log.assign(trace_rows, 0, &value.0, i);
        }

        // Log accumulator
        let mut accumulators = vec![CubicExtension::<F, E>::from(F::ZERO); num_rows];
        let mut value = CubicExtension::<F, E>::ZERO;
        let acc_reg = range_data.row_accumulators;
        let log_lookup_next = range_data.log_lookup_accumulator.register().next();
        for i in 0..num_rows {
            for (k, acc) in (range_idx.0..range_idx.1).step_by(2).zip(acc_reg.iter()) {
                let beta_minus_a = beta - trace_rows[i][k].into();
                let beta_minus_b = beta - trace_rows[i][k + 1].into();
                accumulators[i] += CubicExtension::from(beta_minus_a).inverse()
                    + CubicExtension::from(beta_minus_b).inverse();
                acc.register().assign(trace_rows, 0, &accumulators[i].0, i);
            }

            if i != num_rows - 1 {
                value += accumulators[i] - mult_table_log_entries[i];
                log_lookup_next.assign(trace_rows, 0, &value.0, i);
            }
        }

        // for (i, (acc, mult_table)) in accumulators.into_iter().zip(mult_table_log_entries.into_iter()).enumerate().filter(|(i, _)| *i != num_rows-1)  {
        //     let log_lookup_next = range_data.log_lookup_accumulator.register().next();
        //     value += accumulators[i] - mult_table_log_entries[i];
        //     log_lookup_next.assign(trace_rows, 0, &value.0, i);
        // }
        Ok(())
    }
}
