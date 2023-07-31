use super::constraint::Constraint;
use super::{AirParameters, Chip};
use crate::air::parser::AirParser;
use crate::air::{AirConstraint, RAir, RoundDatum};

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
    fn round_data(&self) -> Vec<RoundDatum> {
        let total = L::num_columns();
        let execution_trace_length = self.execution_trace_length;
        let extended_trace_length = total - execution_trace_length;

        if extended_trace_length == 0 {
            return vec![RoundDatum::new(
                total,
                (0, self.num_global_values),
                self.num_challenges,
            )];
        }
        vec![
            RoundDatum::new(execution_trace_length, (0, 0), self.num_challenges),
            RoundDatum::new(extended_trace_length, (0, self.num_global_values), 0),
        ]
    }

    fn num_public_inputs(&self) -> usize {
        self.num_public_inputs
    }

    fn width(&self) -> usize {
        L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS
    }
}
