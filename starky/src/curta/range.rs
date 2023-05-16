//! Range checks based on logarithmic derivatives.
//!

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::builder::StarkBuilder;
use super::chip::StarkParameters;
use super::constraint::arithmetic::ArithmeticExpression;
use super::extension::cubic::cubic_expression::CubicExpression;
use super::extension::cubic::register::CubicElementRegister;
use super::extension::cubic::{CubicExtension, CubicParameters};
use super::register::{ArrayRegister, ElementRegister, Register, RegisterSerializable};
use super::trace::TraceGenerator;
use crate::curta::register::MemorySlice;

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

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub(crate) fn arithmetic_range_checks<E: CubicParameters<F>>(&mut self) {
        // assign registers for table, multiplicity and accumulators
        let table = self.alloc::<ElementRegister>();
        self.write_data(&table).unwrap();
        let multiplicity = self.alloc::<ElementRegister>();
        self.write_data(&multiplicity).unwrap();
        let multiplicity_table_log = self.alloc::<CubicElementRegister>();
        self.write_data(&multiplicity_table_log).unwrap();
        let log_lookup_accumulator = self.alloc::<CubicElementRegister>();
        self.write_data(&log_lookup_accumulator).unwrap();

        let one = || -> ArithmeticExpression<F, D> { ArithmeticExpression::one() };
        let zero = || -> ArithmeticExpression<F, D> { ArithmeticExpression::zero() };

        let beta = CubicExpression::from_constants([
            F::from_canonical_u64(BETAS[0]),
            F::from_canonical_u64(BETAS[1]),
            F::from_canonical_u64(BETAS[2]),
        ]);

        // Table constraints
        self.assert_expressions_equal_first_row(table.expr(), zero());
        self.assert_expressions_equal_transition(table.expr() + one(), table.next().expr());

        // Multiplicity inverse constraint
        // multiplicity_table_log = multiplicity / (beta - table))
        self.assert_cubic_expressions_equal(
            CubicExpression::from(multiplicity.expr()),
            multiplicity_table_log.extension_expr() * (beta.clone() - table.expr().into()),
        );

        assert_eq!(
            L::NUM_ARITHMETIC_COLUMNS % 2,
            0,
            "The number of arithmetic columns must be even"
        );
        let num_accumulators = L::NUM_ARITHMETIC_COLUMNS / 2;
        let row_accumulators = self.alloc_array::<CubicElementRegister>(num_accumulators);

        let mut row_acc_iter = row_accumulators.iter().map(|r| r.extension_expr());

        let range_idx = L::NUM_FREE_COLUMNS..(L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS);
        let mut range_pairs = range_idx
            .step_by(2)
            .map(|k| (MemorySlice::Local(k, 1), MemorySlice::Local(k + 1, 1)))
            .map(|(a, b)| {
                (
                    beta.clone() - a.expr().into(),
                    beta.clone() - b.expr().into(),
                )
            });

        let ((beta_minus_a_0, beta_minus_b_0), acc_0) =
            (range_pairs.next().unwrap(), row_acc_iter.next().unwrap());

        self.assert_cubic_expressions_equal(
            beta_minus_a_0.clone() + beta_minus_b_0.clone(),
            acc_0.clone() * beta_minus_a_0 * beta_minus_b_0,
        );

        let mut prev = acc_0;
        for ((beta_minus_a, beta_minus_b), acc) in range_pairs.zip(row_acc_iter) {
            self.assert_cubic_expressions_equal(
                beta_minus_a.clone() + beta_minus_b.clone(),
                (acc.clone() - prev.clone()) * beta_minus_a * beta_minus_b,
            );
            prev = acc;
        }

        self.assert_cubic_expressions_equal_transition(
            log_lookup_accumulator.extension_expr() + prev.clone()
                - multiplicity_table_log.extension_expr(),
            log_lookup_accumulator.next().extension_expr(),
        );

        self.assert_cubic_expressions_equal_first_row(
            log_lookup_accumulator.extension_expr(),
            CubicExpression::from_constants([F::ZERO; 3]),
        );

        self.assert_cubic_expressions_equal_last_row(
            log_lookup_accumulator.extension_expr() + prev
                - multiplicity_table_log.extension_expr(),
            CubicExpression::from_constants([F::ZERO; 3]),
        );

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

        let mut sum_check = CubicExtension::<F, E>::ZERO;
        let one_ext = CubicExtension::<F, E>::ONE;
        for row in trace_rows.iter() {
            for value in row[range_idx.0..range_idx.1].iter() {
                sum_check += one_ext / (beta - (*value).into());
                let value = value.to_canonical_u64() as usize;
                assert!(value < 1 << 16);
                multiplicities[value] += F::ONE;
            }
        }

        // write multiplicities into the trace
        let multiplicity = range_data.multiplicity.register();
        for i in 0..num_rows {
            multiplicity.assign(trace_rows, 0, &mut vec![multiplicities[i]], i);
        }

        // write multiplicity inverse constraint
        let mult_table_log_entries = multiplicities
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let table = CubicExtension::from(F::from_canonical_usize(i));
                CubicExtension::from(*x) / (beta - table)
            })
            .collect::<Vec<_>>();

        let log_sum = mult_table_log_entries.iter().sum::<CubicExtension<F, E>>();
        assert_eq!(log_sum, sum_check, "Sum check failed");

        let mult_table_log = range_data.multiplicity_table_log.register();
        for (i, value) in mult_table_log_entries.iter().enumerate() {
            mult_table_log.assign(trace_rows, 0, &value.0, i);
        }

        // Log accumulator
        let mut accumulators = vec![CubicExtension::<F, E>::from(F::ZERO); num_rows];
        let mut sum_acc_check = CubicExtension::<F, E>::ZERO;
        let mut value = CubicExtension::<F, E>::ZERO;
        let acc_reg = range_data.row_accumulators;
        for i in 0..num_rows {
            for (k, acc) in (range_idx.0..range_idx.1).step_by(2).zip(acc_reg.iter()) {
                let beta_minus_a = beta - trace_rows[i][k].into();
                let beta_minus_b = beta - trace_rows[i][k + 1].into();
                accumulators[i] += CubicExtension::from(beta_minus_a).inverse()
                    + CubicExtension::from(beta_minus_b).inverse();
                acc.register().assign(trace_rows, 0, &accumulators[i].0, i);
            }

            sum_acc_check += accumulators[i];

            if i != num_rows - 1 {
                let log_lookup_next = range_data.log_lookup_accumulator.register().next();
                // let value = accumulators[i] - mult_table_log_entries[i];
                value += accumulators[i] - mult_table_log_entries[i];
                log_lookup_next.assign(trace_rows, 0, &value.0, i);
            }

            if i == num_rows - 1 {
                assert_eq!(sum_acc_check, sum_check, "accumulator check failed");
                assert_eq!(
                    value + accumulators[i] - mult_table_log_entries[i],
                    CubicExtension::ZERO,
                    "value check failed"
                );
            }
        }

        Ok(())
    }
}
