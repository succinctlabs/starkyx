use super::constraint::Constraint;
use super::{AirParameters, Chip};
use crate::air::parser::AirParser;
use crate::air::{AirConstraint, RAir};

impl<AP: AirParser, L: AirParameters<Field = AP::Field>> RAir<AP> for Chip<L>
where
    Constraint<L>: AirConstraint<AP>,
{
    /// Evaluation of the vanishing polynomials.
    fn eval(&self, parser: &mut AP) {
        for constraint in self.constraints.iter() {
            constraint.eval(parser);
        }
    }

    /// The maximal constraint degree
    fn constraint_degree(&self) -> usize {
        3
    }

    /// Columns for each round
    fn round_lengths(&self) -> Vec<usize> {
        let total = L::num_columns();
        let execution_trace_length = self.execution_trace_length;
        let execution_trace_length = total - execution_trace_length;
        if execution_trace_length == 0 {
            return vec![total];
        }
        vec![execution_trace_length, total - execution_trace_length]
    }

    /// The number of challenges after each round
    fn num_challenges(&self, round: usize) -> usize {
        match round {
            0 => self.num_challenges,
            _ => 0,
        }
    }
    fn width(&self) -> usize {
        L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS
    }
}
