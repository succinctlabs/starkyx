use crate::air::parser::AirParser;
use crate::air::Air;
use crate::challenger::Challenger;

pub trait StarkConfig: 'static {
    type Parser: AirParser;
    type Air: Air<Self::Parser>;

    type Challenger: Challenger<Self::Parser>;

    /// Columns for each round
    fn round_lengths(&self) -> &[usize];

    /// The number of challenges per round
    fn num_challenges(&self, round: usize) -> usize;

    fn constraint_degree(&self) -> usize;
}
