//! This module implements a lookup argument based on the logarithmic derivative as in
//! https://eprint.iacr.org/2022/1530.pdf
//!
//! The basic idea

use alloc::collections::VecDeque;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::Lookup;
use crate::curta::air::parser::AirParser;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::constraint::arithmetic::ArithmeticExpression;
use crate::curta::extension::cubic::element::CubicElement;
use crate::curta::extension::cubic::gadget::CubicGadget;
use crate::curta::extension::cubic::parser::CubicParser;
use crate::curta::extension::cubic::register::CubicElementRegister;
use crate::curta::register::{
    ArrayRegister, ElementRegister, MemorySlice, Register, RegisterSerializable,
};
use crate::curta::stark::vars as new_vars;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone)]
pub struct LogLookup {
    pub(crate) challenge_idx: usize,
    pub(crate) table: ElementRegister,
    pub(crate) values: ArrayRegister<ElementRegister>,
    pub(crate) multiplicity: ElementRegister,
    pub(crate) multiplicity_table_log: CubicElementRegister,
    pub(crate) row_accumulators: ArrayRegister<CubicElementRegister>,
    pub(crate) log_lookup_accumulator: CubicElementRegister,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitData {
    mid: usize,
    values_range: (usize, usize),
    acc_range: (usize, usize),
}

impl SplitData {
    pub fn new(mid: usize, values_range: (usize, usize), acc_range: (usize, usize)) -> Self {
        Self {
            mid,
            values_range,
            acc_range,
        }
    }
    pub(crate) fn split<'a, T>(&self, trace_row: &'a mut [T]) -> (&'a [T], &'a mut [T]) {
        let (left, right) = trace_row.split_at_mut(self.mid);
        (
            &left[self.values_range.0..self.values_range.1],
            &mut right[self.acc_range.0..self.acc_range.1],
        )
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
        assert!(
            values_idx.0 < acc_idx.0,
            "Illegal memory pattern, expected values indices \
        to be to the right of accumulator indices, \
        instead got: values_idx: {:?}, acc_idx: {:?}",
            values_idx,
            acc_idx
        );
        SplitData::new(
            values_idx.1,
            (values_idx.0, values_idx.1),
            (acc_idx.0 - values_idx.1, acc_idx.1 - values_idx.1),
        )
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
        betas: &[F],
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let b_idx = 3 * self.challenge_idx;
        let beta_array = &betas[b_idx..b_idx + 3];
        let beta = CubicElement([
            P::from(FE::from_basefield(beta_array[0])),
            P::from(FE::from_basefield(beta_array[1])),
            P::from(FE::from_basefield(beta_array[2])),
        ]);

        let multiplicity = CubicElement::from_base(
            self.multiplicity.register().packed_generic_vars(vars)[0],
            P::ZEROS,
        );
        let table =
            CubicElement::from_base(self.table.register().packed_generic_vars(vars)[0], P::ZEROS);
        let multiplicity_table_log = CubicElement::from_slice(
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
            .map(|r| CubicElement::from_slice(r.register().packed_generic_vars(vars)));

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
                    CubicElement::from_base(a, P::ZEROS),
                    CubicElement::from_base(b, P::ZEROS),
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

        let log_lookup_accumulator = CubicElement::from_slice(
            self.log_lookup_accumulator
                .register()
                .packed_generic_vars(vars),
        );
        let log_lookup_accumulator_next = CubicElement::from_slice(
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
        betas: &[Target],
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let b_idx = 3 * self.challenge_idx;
        let beta_slice = &betas[b_idx..b_idx + 3];
        let beta = CubicElement([
            builder.convert_to_ext(beta_slice[0]),
            builder.convert_to_ext(beta_slice[1]),
            builder.convert_to_ext(beta_slice[2]),
        ]);

        let multiplicity = CubicGadget::from_base_extension(
            builder,
            self.multiplicity.register().ext_circuit_vars(vars)[0],
        );

        let table = CubicGadget::from_base_extension(
            builder,
            self.table.register().ext_circuit_vars(vars)[0],
        );
        let multiplicity_table_log = CubicElement::from_slice(
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
            .map(|r| CubicElement::from_slice(r.register().ext_circuit_vars(vars)))
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

        let log_lookup_accumulator = CubicElement::from_slice(
            self.log_lookup_accumulator
                .register()
                .ext_circuit_vars(vars),
        );
        let log_lookup_accumulator_next = CubicElement::from_slice(
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

    pub fn packed_generic_constraints_new<
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
    >(
        &self,
        vars: new_vars::StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }, { CHALLENGES }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let b_idx = 3 * self.challenge_idx;
        let beta_array = &vars.challenges[b_idx..b_idx + 3];
        let beta = CubicElement([
            P::from(beta_array[0]),
            P::from(beta_array[1]),
            P::from(beta_array[2]),
        ]);

        let multiplicity = CubicElement::from_base(
            self.multiplicity.register().packed_generic_vars_new(vars)[0],
            P::ZEROS,
        );
        let table = CubicElement::from_base(
            self.table.register().packed_generic_vars_new(vars)[0],
            P::ZEROS,
        );
        let multiplicity_table_log = CubicElement::from_slice(
            self.multiplicity_table_log
                .register()
                .packed_generic_vars_new(vars),
        );

        let mult_table_constraint = multiplicity - multiplicity_table_log * (beta - table);
        for consr in mult_table_constraint.0 {
            yield_constr.constraint(consr);
        }

        let mut row_acc_iter = self
            .row_accumulators
            .iter()
            .map(|r| CubicElement::from_slice(r.register().packed_generic_vars_new(vars)));

        let mut values_pairs = (0..self.values.len())
            .step_by(2)
            .map(|k| {
                (
                    self.values.get(k).register().packed_generic_vars_new(vars)[0],
                    self.values
                        .get(k + 1)
                        .register()
                        .packed_generic_vars_new(vars)[0],
                )
            })
            .map(|(a, b)| {
                (
                    CubicElement::from_base(a, P::ZEROS),
                    CubicElement::from_base(b, P::ZEROS),
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

        let log_lookup_accumulator = CubicElement::from_slice(
            self.log_lookup_accumulator
                .register()
                .packed_generic_vars_new(vars),
        );
        let log_lookup_accumulator_next = CubicElement::from_slice(
            self.log_lookup_accumulator
                .next()
                .register()
                .packed_generic_vars_new(vars),
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

    pub fn ext_circuit_constraints_new<
        F: RichField + Extendable<D>,
        const D: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
        const CHALLENGES: usize,
    >(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: new_vars::StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }, { CHALLENGES }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let b_idx = 3 * self.challenge_idx;
        let beta_slice = &vars.challenges[b_idx..b_idx + 3];
        let beta = CubicElement([beta_slice[0], beta_slice[1], beta_slice[2]]);

        let multiplicity = CubicGadget::from_base_extension(
            builder,
            self.multiplicity.register().ext_circuit_vars_new(vars)[0],
        );

        let table = CubicGadget::from_base_extension(
            builder,
            self.table.register().ext_circuit_vars_new(vars)[0],
        );
        let multiplicity_table_log = CubicElement::from_slice(
            self.multiplicity_table_log
                .register()
                .ext_circuit_vars_new(vars),
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
            .map(|r| CubicElement::from_slice(r.register().ext_circuit_vars_new(vars)))
            .collect::<VecDeque<_>>();

        let mut range_pairs = (0..self.values.len())
            .step_by(2)
            .map(|k| {
                (
                    self.values.get(k).register().ext_circuit_vars_new(vars)[0],
                    self.values.get(k + 1).register().ext_circuit_vars_new(vars)[0],
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

        let log_lookup_accumulator = CubicElement::from_slice(
            self.log_lookup_accumulator
                .register()
                .ext_circuit_vars_new(vars),
        );
        let log_lookup_accumulator_next = CubicElement::from_slice(
            self.log_lookup_accumulator
                .next()
                .register()
                .ext_circuit_vars_new(vars),
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

    pub fn eval<AP: AirParser>(&self, parser: &mut AP) {
        let mut cubic_parser = CubicParser::new(parser);

        let b_idx = 3 * self.challenge_idx;
        let beta_slice = &cubic_parser.parser.challenge_slice()[b_idx..b_idx + 3];
        let beta = CubicElement([beta_slice[0], beta_slice[1], beta_slice[2]]);

        let multiplicity =
            cubic_parser.from_base(self.multiplicity.register().eval_slice(cubic_parser.parser)[0]);

        let table =
            cubic_parser.from_base(self.table.register().eval_slice(cubic_parser.parser)[0]);
        let multiplicity_table_log = CubicElement::from_slice(
            self.multiplicity_table_log
                .register()
                .eval_slice(cubic_parser.parser),
        );

        let beta_minus_table = cubic_parser.sub(&beta, &table);
        let mut mult_table_constraint =
            cubic_parser.mul(&multiplicity_table_log, &beta_minus_table);
        mult_table_constraint = cubic_parser.sub(&multiplicity, &mult_table_constraint);
        for consr in mult_table_constraint.0 {
            cubic_parser.parser.constraint(consr);
        }

        let mut row_acc_vec = self
            .row_accumulators
            .iter()
            .map(|r| CubicElement::from_slice(r.register().eval_slice(cubic_parser.parser)))
            .collect::<VecDeque<_>>();

        let mut range_pairs = (0..self.values.len())
            .step_by(2)
            .map(|k| {
                let a_base = self
                    .values
                    .get(k)
                    .register()
                    .eval_slice(cubic_parser.parser)[0];
                let b_base = self
                    .values
                    .get(k + 1)
                    .register()
                    .eval_slice(cubic_parser.parser)[0];
                let a = cubic_parser.from_base(a_base);
                let b = cubic_parser.from_base(b_base);
                (a, b)
            })
            .collect::<Vec<_>>()
            .iter()
            .map(|(a, b)| (cubic_parser.sub(&beta, a), cubic_parser.sub(&beta, b)))
            .collect::<VecDeque<_>>();

        let ((beta_minus_a_0, beta_minus_b_0), acc_0) = (
            range_pairs.pop_front().unwrap(),
            row_acc_vec.pop_front().unwrap(),
        );

        let beta_minus_a_b = cubic_parser.mul(&beta_minus_a_0, &beta_minus_b_0);
        let acc_beta_m_ab = cubic_parser.mul(&acc_0, &beta_minus_a_b);
        let mut constr_0 = cubic_parser.add(&beta_minus_a_0, &beta_minus_b_0);
        constr_0 = cubic_parser.sub(&constr_0, &acc_beta_m_ab);
        for consr in constr_0.0 {
            cubic_parser.parser.constraint(consr);
        }

        let mut prev = acc_0;
        for ((beta_minus_a, beta_minus_b), acc) in range_pairs.iter().zip(row_acc_vec.iter()) {
            let acc_minus_prev = cubic_parser.sub(acc, &prev);
            let mut product = cubic_parser.mul(beta_minus_a, &beta_minus_b);
            product = cubic_parser.mul(&product, &acc_minus_prev);
            let mut constraint = cubic_parser.add(&beta_minus_a, &beta_minus_b);
            constraint = cubic_parser.sub(&constraint, &product);
            for consr in constraint.0 {
                cubic_parser.parser.constraint(consr);
            }
            prev = *acc;
        }

        let log_lookup_accumulator = CubicElement::from_slice(
            self.log_lookup_accumulator
                .register()
                .eval_slice(cubic_parser.parser),
        );
        let log_lookup_accumulator_next = CubicElement::from_slice(
            self.log_lookup_accumulator
                .next()
                .register()
                .eval_slice(cubic_parser.parser),
        );

        let mut acc_transition_constraint =
            cubic_parser.sub(&log_lookup_accumulator_next, &log_lookup_accumulator);
        acc_transition_constraint = cubic_parser.sub(&acc_transition_constraint, &prev);
        acc_transition_constraint =
            cubic_parser.add(&acc_transition_constraint, &multiplicity_table_log);
        for consr in acc_transition_constraint.0 {
            cubic_parser.parser.constraint_transition(consr);
        }

        let acc_first_row_constraint = log_lookup_accumulator;
        for consr in acc_first_row_constraint.0 {
            cubic_parser.parser.constraint_first_row(consr);
        }

        let mut acc_last_row_constraint = cubic_parser.add(&log_lookup_accumulator, &prev);
        acc_last_row_constraint =
            cubic_parser.sub(&acc_last_row_constraint, &multiplicity_table_log);
        for consr in acc_last_row_constraint.0 {
            cubic_parser.parser.constraint_last_row(consr);
        }
    }
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub(crate) fn lookup_log_derivative(
        &mut self,
        table: &ElementRegister,
        values: &ArrayRegister<ElementRegister>,
    ) {
        // assign registers for table, multiplicity and accumulators
        let multiplicity = self.alloc::<ElementRegister>();
        let multiplicity_table_log = self.alloc::<CubicElementRegister>();
        let log_lookup_accumulator = self.alloc::<CubicElementRegister>();
        let row_accumulators = self.alloc_array::<CubicElementRegister>(values.len() / 2);

        let challenge_idx = self.num_verifier_challenges;
        self.num_verifier_challenges += 1;

        self.range_data = Some(Lookup::LogDerivative(LogLookup {
            challenge_idx,
            table: *table,
            values: *values,
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
            0,
            L::NUM_ARITHMETIC_COLUMNS,
            // L::NUM_FREE_COLUMNS,
            // L::NUM_ARITHMETIC_COLUMNS,
        ));

        self.lookup_log_derivative(&table, &values)
    }
}
