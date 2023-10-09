use core::marker::PhantomData;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::entry::{LogEntry, LogEntryValue};
use crate::air::extension::cubic::CubicParser;
use crate::chip::constraint::kind::ConstraintKind;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::cubic::element::CubicElement;
use crate::math::prelude::CubicParameters;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LogConstraints<AP, E>(pub PhantomData<(AP, E)>);

impl<AP: CubicParser<E>, E: CubicParameters<AP::Field>> LogConstraints<AP, E> {
    /// Computes the constraint for `mult_a/(beta - a) + mult_b/(beta - b) = result`.
    ///
    /// This rsulting constraint is of degree 3 and assumes that both `beta-a` and `beta-b` are
    /// different from zero.
    ///
    /// Under the assumption that both `beta-a` and `beta-b` are different from zero, the equation
    ///  `mult_a/(beta - a) + mult_b/(beta - b) = result` is equivalent to the degree 3 constraint:
    ///  `mult_a * (beta - b) + mult_b * (beta - a) = result * (beta - a) * (beta - b)`
    #[inline]
    pub fn log_arithmetic(
        parser: &mut AP,
        beta: CubicElement<AP::Var>,
        a: LogEntryValue<AP>,
        b: LogEntryValue<AP>,
        result: CubicElement<AP::Var>,
    ) -> CubicElement<AP::Var> {
        let a_value = a.value;
        let b_value = b.value;
        let mult_a = a.multiplier;
        let mult_b = b.multiplier;
        let beta_minus_a = parser.sub_extension(beta, a_value);
        let beta_minus_b = parser.sub_extension(beta, b_value);

        let mut rhs = parser.mul_extension(beta_minus_a, beta_minus_b);
        rhs = parser.mul_extension(rhs, result);

        let mult_a_beta_minus_b = parser.scalar_mul_extension(beta_minus_b, mult_a);
        let mult_b_beta_minus_a = parser.scalar_mul_extension(beta_minus_a, mult_b);
        let lhs = parser.add_extension(mult_a_beta_minus_b, mult_b_beta_minus_a);

        parser.sub_extension(lhs, rhs)
    }

    /// Computes the constraint for `m/(beta - a) = result`.
    ///
    /// This rsulting constraint is of degree 2 and assumes that `beta-a` is different from zero.
    ///
    /// Under the assumption that `beta-a` is non-zero, the equation `m/(beta - a) = result` is
    /// equivalent to the degree 2 constraint: `m = result * (beta - a)`.
    #[inline]
    pub fn log(
        parser: &mut AP,
        beta: CubicElement<AP::Var>,
        a: LogEntryValue<AP>,
        result: CubicElement<AP::Var>,
    ) -> CubicElement<AP::Var> {
        let beta_minus_a = parser.sub_extension(beta, a.value);
        let rhs = parser.mul_extension(beta_minus_a, result);
        let m_ext = parser.element_from_base_field(a.multiplier);
        parser.sub_extension(m_ext, rhs)
    }

    #[inline]
    pub fn log_row_accumulation<'a, T: EvalCubic>(
        parser: &mut AP,
        beta: CubicElement<AP::Var>,
        entries: &'a [LogEntry<T>],
        intermediate_values: ArrayRegister<CubicRegister>,
        result: CubicElement<AP::Var>,
    ) -> Option<&'a LogEntry<T>> {
        let entry_chunks = entries.chunks_exact(2);
        let last_element = entry_chunks.remainder().first();

        let zero = parser.zero_extension();
        let mut prev = zero;
        for (chunk, row_acc) in entry_chunks.zip_eq(intermediate_values) {
            let a = chunk[0].eval(parser);
            let b = chunk[1].eval(parser);
            let acc = row_acc.eval(parser);
            let acc_minus_prev = parser.sub_extension(acc, prev);
            let constraint = LogConstraints::log_arithmetic(parser, beta, a, b, acc_minus_prev);
            parser.constraint_extension(constraint);
            prev = acc;
        }

        last_element

        // match kind {
        //     ConstraintKind::Trace => {
        //         let accumulated_value_pair_chunks_next = intermediate_values
        //         .last()
        //         .map_or(zero, |r| r.next().eval(parser));
        //         let mut acc_transition_constraint = parser.sub_extension(result, accumulated_value_pair_chunks_next);
        //     }
        //     ConstraintKind::Global => {}
        // }

        // let accumulated_value_pair_chunks = intermediate_values
        //     .last()
        //     .map_or(zero, |r| eval_last(r, parser));

        // let eval_last_entry = |r: &LogEntry<T>, parser: &mut AP| match kind {
        //     ConstraintKind::Transition => r.next().eval(parser),
        //     _ => r.eval(parser),
        // };

        // let mut acc_constraint = parser.sub_extension(result, accumulated_value_pair_chunks);

        // if let Some(last) = last_element {
        //     let a = eval_last_entry(last, parser);
        //     acc_constraint = LogConstraints::log(parser, beta, a, acc_constraint);
        // }

        // acc_constraint
    }

    #[inline]
    pub fn log_trace_accumulation<T: EvalCubic>(
        parser: &mut AP,
        beta: CubicElement<AP::Var>,
        entries: &[LogEntry<T>],
        intermediate_values: ArrayRegister<CubicRegister>,
        trace_accumulator: CubicElement<AP::Var>,
    ) {}

    #[inline]
    pub fn log_global_accumulation<T: EvalCubic>(
        parser: &mut AP,
        beta: CubicElement<AP::Var>,
        entries: &[LogEntry<T>],
        intermediate_values: ArrayRegister<CubicRegister>,
        global_accumulator: CubicElement<AP::Var>,
    ) {}
}
