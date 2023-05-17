use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::cubic_expression::CubicExpression;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::constraint::Constraint;

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in the first row.
    pub fn assert_cubic_expressions_equal_first_row(
        &mut self,
        a: CubicExpression<F, D>,
        b: CubicExpression<F, D>,
    ) {
        let a_array = a.into_expressions_array();
        let b_array = b.into_expressions_array();

        for (a, b) in a_array.into_iter().zip(b_array.into_iter()) {
            let constraint = Constraint::First((a - b).into());
            self.constraints.push(constraint);
        }
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in the last row.
    pub fn assert_cubic_expressions_equal_last_row(
        &mut self,
        a: CubicExpression<F, D>,
        b: CubicExpression<F, D>,
    ) {
        let a_array = a.into_expressions_array();
        let b_array = b.into_expressions_array();

        for (a, b) in a_array.into_iter().zip(b_array.into_iter()) {
            let constraint = Constraint::Last((a - b).into());
            self.constraints.push(constraint);
        }
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in all rows but the last (useful for when dealing with registers between the local and
    /// next row).
    pub fn assert_cubic_expressions_equal_transition(
        &mut self,
        a: CubicExpression<F, D>,
        b: CubicExpression<F, D>,
    ) {
        let a_array = a.into_expressions_array();
        let b_array = b.into_expressions_array();

        for (a, b) in a_array.into_iter().zip(b_array.into_iter()) {
            let constraint = Constraint::Transition((a - b).into());
            self.constraints.push(constraint);
        }
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in all rows.
    pub fn assert_cubic_expressions_equal(
        &mut self,
        a: CubicExpression<F, D>,
        b: CubicExpression<F, D>,
    ) {
        let a_array = a.into_expressions_array();
        let b_array = b.into_expressions_array();

        for (a, b) in a_array.into_iter().zip(b_array.into_iter()) {
            let constraint = Constraint::All((a - b).into());
            self.constraints.push(constraint);
        }
    }
}
