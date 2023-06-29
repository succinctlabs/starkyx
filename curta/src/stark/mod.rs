//! A STARK with randomized AIR.
//!
//!

use self::config::StarkConfig;
use crate::air::parser::AirParser;
use crate::air::Air;

pub mod config;
pub mod proof;
pub mod prover;
pub mod verifier;

pub trait Stark<AP: AirParser, SC: StarkConfig<AP>> {
    type Air: Air<AP>;

    fn air(&self) -> &Self::Air;

    /// Columns for each round
    fn round_lengths(&self) -> &[usize];

    /// The number of challenges per round
    fn num_challenges(&self, round: usize) -> usize;
}
