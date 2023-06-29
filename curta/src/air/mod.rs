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
}
