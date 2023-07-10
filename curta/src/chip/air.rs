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
        todo!()
    }

    /// Columns for each round
    fn round_lengths(&self) -> Vec<usize> {
        vec![L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS]
    }

    /// The number of challenges per round
    fn num_challenges(&self, _round: usize) -> usize {
        0
    }
    fn width(&self) -> usize {
        L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS
    }
}
