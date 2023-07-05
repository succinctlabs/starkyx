pub mod opening;
pub mod parser;

#[cfg(test)]
pub mod fibonacci;

use parser::AirParser;

pub trait Air<AP: AirParser> {
    /// Evaluation of the vanishing polynomials.
    fn eval(&self, parser: &mut AP);

    /// The maximal constraint degree
    fn constraint_degree(&self) -> usize;

    fn width(&self) -> usize;
}

pub trait RAir<AP: AirParser> {
    /// Evaluation of the vanishing polynomials.
    fn eval(&self, parser: &mut AP);

    /// The maximal constraint degree
    fn constraint_degree(&self) -> usize;

    /// Columns for each round
    fn round_lengths(&self) -> Vec<usize>;

    /// The number of challenges per round
    fn num_challenges(&self, round: usize) -> usize;

    fn num_rounds(&self) -> usize {
        self.round_lengths().len()
    }

    fn quotient_degree_factor(&self) -> usize {
        1.max(self.constraint_degree() - 1)
    }
}

impl<AP: AirParser, T: Air<AP>> RAir<AP> for T {
    fn eval(&self, parser: &mut AP) {
        Air::eval(self, parser)
    }

    fn constraint_degree(&self) -> usize {
        Air::constraint_degree(self)
    }

    fn round_lengths(&self) -> Vec<usize> {
        vec![self.width()]
    }

    fn num_challenges(&self, _round: usize) -> usize {
        0
    }
}
