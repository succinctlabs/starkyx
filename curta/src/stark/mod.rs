//! A STARK with randomized AIR.
//!
//!

use self::config::StarkConfig;
use crate::air::parser::AirParser;
use crate::air::RAir;

pub mod config;
pub trait Stark<AP: AirParser, SC: StarkConfig<AP>> {
    type Air: RAir<AP>;

    fn air(&self) -> &Self::Air;
}
