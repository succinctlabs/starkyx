use alloc::collections::VecDeque;

use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2_maybe_rayon::*;

use super::Lookup;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::constraint::arithmetic::ArithmeticExpression;
use crate::curta::extension::cubic::array::CubicArray;
use crate::curta::extension::cubic::gadget::CubicGadget;
use crate::curta::extension::cubic::register::CubicElementRegister;
use crate::curta::extension::cubic::{CubicExtension, CubicParameters};
use crate::curta::register::{
    ArrayRegister, ElementRegister, MemorySlice, Register, RegisterSerializable,
};
use crate::curta::trace::TraceGenerator;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

const BETAS: [u64; 3] = [
    17800306513594245228,
    422882772345461752,
    14491510587541603695,
];

#[derive(Debug, Clone)]
pub struct LogLookup {
    table: ElementRegister,
    values: ArrayRegister<ElementRegister>,
    multiplicity: ElementRegister,
    multiplicity_table_log: CubicElementRegister,
    row_accumulators: ArrayRegister<CubicElementRegister>,
    log_lookup_accumulator: CubicElementRegister,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitData {
    mid : usize, 
    switch : bool,
    values_range : (usize, usize),
    acc_range : (usize, usize),
}

impl SplitData {
    pub fn new(mid : usize, switch: bool, values_range : (usize, usize), acc_range : (usize, usize)) -> Self {
        Self{
            mid,
            switch,
            values_range,
            acc_range,
        }
    }
    pub(crate) fn split_row<'a, T>(&self, trace_row : &'a mut [T]) -> (&'a [T], &'a mut [T]) {
        let (left, right) = trace_row.split_at_mut(self.mid);
        if self.switch {
        (&right[self.values_range.0..self.values_range.1], &mut left[self.acc_range.0..self.acc_range.1])
        }
        else {
            (&left[self.values_range.0..self.values_range.1], &mut right[self.acc_range.0..self.acc_range.1])
        }
    }
}

impl LogLookup {

    #[inline]
    pub(crate) fn values_idx(&self) -> (usize, usize) {
        self.values.register().get_range()
    }

    pub(crate) fn split_data(&self) -> SplitData {
        let values_idx = self.values.register().get_range();
        let acc_idx = self.row_accumulators.register().get_range(); 
        if values_idx.0 > acc_idx.0 {
            SplitData::new(acc_idx.1, true, 
                (values_idx.0-acc_idx.1, values_idx.1-acc_idx.1),
                (acc_idx.0, acc_idx.1)
            )
        } else {
            SplitData::new(values_idx.1, false, (values_idx.0, values_idx.1), (acc_idx.0-values_idx.1, acc_idx.1-values_idx.1)) 
        }
    }

    pub(crate) fn split_row<'a, T>(&self, trace_row : &'a mut [T]) -> (&'a [T], &'a mut [T]) {
        assert!(self.values.register().index() > self.log_lookup_accumulator.register().index());
        let values_idx = self.values_idx();
        let acc_idx = self.row_accumulators.register().get_range();
        let (accumulators, values) = trace_row.split_at_mut(acc_idx.1);
        (&values[values_idx.0-acc_idx.1..values_idx.1-acc_idx.1], &mut accumulators[acc_idx.0..acc_idx.1])
    }
     
    pub fn packed_generic_constraints<
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

        let mut values_pairs = (0..self.values.len())
            .step_by(2)
            .map(|k| {
                (
                    self.values.get(k).register().packed_generic_vars(vars)[0],
                    self.values.get(k + 1).register().packed_generic_vars(vars)[0],
                )
            })
            .map(|(a, b)| {
                (
                    CubicArray::from_base(a, P::ZEROS),
                    CubicArray::from_base(b, P::ZEROS),
                )
            })
            .map(|(a, b)| (beta - a, beta - b));

        let ((beta_minus_a_0, beta_minus_b_0), acc_0) =
            (values_pairs.next().unwrap(), row_acc_iter.next().unwrap());

        let constr_0 = beta_minus_a_0 + beta_minus_b_0 - acc_0 * beta_minus_a_0 * beta_minus_b_0;
        for consr in constr_0.0 {
            yield_constr.constraint(consr);
        }

        let mut prev = acc_0;
        for ((beta_minus_a, beta_minus_b), acc) in values_pairs.zip(row_acc_iter) {
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

        let mut range_pairs = (0..self.values.len())
            .step_by(2)
            .map(|k| {
                (
                    self.values.get(k).register().ext_circuit_vars(vars)[0],
                    self.values.get(k + 1).register().ext_circuit_vars(vars)[0],
                )
            })
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

        let ((beta_minus_a_0, beta_minus_b_0), acc_0) = (
            range_pairs.pop_front().unwrap(),
            row_acc_vec.pop_front().unwrap(),
        );

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

    pub(crate) fn lookup_log_derivative(&mut self, table : &ElementRegister, values : &ArrayRegister<ElementRegister>) {
        // assign registers for table, multiplicity and accumulators
        let multiplicity = self.alloc::<ElementRegister>();
        let multiplicity_table_log = self.alloc::<CubicElementRegister>();
        let log_lookup_accumulator = self.alloc::<CubicElementRegister>();
        let row_accumulators = self.alloc_array::<CubicElementRegister>(values.len() / 2);

        self.range_data = Some(Lookup::LogDerivative(LogLookup {
            table: *table,
            values : *values,
            multiplicity,
            multiplicity_table_log,
            row_accumulators,
            log_lookup_accumulator,
        }));
    }

    pub(crate) fn arithmetic_range_checks(&mut self) {
        // assign registers for table, multiplicity and accumulators
        let table = self.alloc::<ElementRegister>();
        self.range_table = Some(table);

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
        let values = ArrayRegister::<ElementRegister>::from_register_unsafe(MemorySlice::Local(
            L::NUM_FREE_COLUMNS,
            L::NUM_ARITHMETIC_COLUMNS,
        ));

        self.lookup_log_derivative(&table, &values)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceGenerator<F, D> {
    pub fn write_lookups<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        trace_rows: &mut Vec<Vec<F>>,
        lookup_data: &LogLookup,
        // range_idx: (usize, usize),
    ) -> Result<()> {
        // Get the challenge
        let betas = [
            F::from_canonical_u64(BETAS[0]),
            F::from_canonical_u64(BETAS[1]),
            F::from_canonical_u64(BETAS[2]),
        ];
        let beta = CubicExtension::<F, E>::from(betas);

        let values_idx = lookup_data.values_idx();

        // Calculate multiplicities
        let mut multiplicities = vec![F::ZERO; num_rows];

        for row in trace_rows.iter() {
            for value in row[values_idx.0..values_idx.1].iter() {
                // sum_check += one_ext / (beta - (*value).into());
                let value = value.to_canonical_u64() as usize;
                assert!(value < 1 << 16);
                multiplicities[value] += F::ONE;
            }
        }

        // Write multiplicities into the trace
        let multiplicity = lookup_data.multiplicity.register();
        for i in 0..num_rows {
            multiplicity.assign(trace_rows, 0, &mut vec![multiplicities[i]], i);
        }

        // Write multiplicity inverse constraint
        let mult_table_log_entries = multiplicities
            .par_iter()
            .enumerate()
            .map(|(i, x)| {
                let table = CubicExtension::from(F::from_canonical_usize(i));
                CubicExtension::from(*x) / (beta - table)
            })
            .collect::<Vec<_>>();

        let mult_table_log = lookup_data.multiplicity_table_log.register();
        for (i, value) in mult_table_log_entries.iter().enumerate() {
            mult_table_log.assign(trace_rows, 0, &value.0, i);
        }

        // Log accumulator
        let mut value = CubicExtension::<F, E>::ZERO;
        let split_data = lookup_data.split_data();
        let accumulators = trace_rows
            .par_iter_mut()
            .map(|row| {
                let (values, accumulators) = split_data.split_row(row);
                let mut accumumulator = CubicExtension::<F, E>::from(F::ZERO);
                for (k, pair) in values.chunks(2).enumerate() {
                    let beta_minus_a = beta - pair[0].into();
                    let beta_minus_b = beta - pair[1].into();
                    accumumulator += CubicExtension::from(beta_minus_a).inverse()
                        + CubicExtension::from(beta_minus_b).inverse();
                    accumulators[3 * k..3 * k + 3].copy_from_slice(&accumumulator.0);
                }
                accumumulator
            })
            .collect::<Vec<_>>();

        let log_lookup_next = lookup_data.log_lookup_accumulator.register().next();
        for (i, (acc, mult_table)) in accumulators
            .into_iter()
            .zip(mult_table_log_entries.into_iter())
            .enumerate()
            .filter(|(i, _)| *i != num_rows - 1)
        {
            value += acc - mult_table;
            log_lookup_next.assign(trace_rows, 0, &value.0, i);
        }
        Ok(())
    }
}
