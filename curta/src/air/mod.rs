pub mod extension;
pub mod opening;
pub mod parser;

#[cfg(test)]
pub mod fibonacci;

use parser::AirParser;

pub trait AirConstraint<AP: AirParser> {
    /// Evaluation of the vanishing polynomials.
    fn eval(&self, parser: &mut AP);
}

pub trait RAir<AP: AirParser> {
    /// Evaluation of the vanishing polynomials.
    fn eval(&self, parser: &mut AP);

    fn width(&self) -> usize;

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
