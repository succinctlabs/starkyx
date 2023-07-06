use super::{AirParameters, Chip};
use crate::air::parser::AirParser;
use crate::air::RAir;

impl<AP: AirParser, L: AirParameters<Field = AP::Field>> RAir<AP> for Chip<L> {
    /// Evaluation of the vanishing polynomials.
    fn eval(&self, parser: &mut AP) {}

    /// The maximal constraint degree
    fn constraint_degree(&self) -> usize {
        todo!()
    }

    /// Columns for each round
    fn round_lengths(&self) -> Vec<usize> {
        todo!()
    }

    /// The number of challenges per round
    fn num_challenges(&self, round: usize) -> usize {
        todo!()
    }
    fn width(&self) -> usize {
        todo!()
    }
}
