//! Lookup table
//!

pub mod log_der;

use log_der::LogLookup;

use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum Lookup<F: Field, E: CubicParameters<F>, const N: usize> {
    LogDerivative(LogLookup<F, E, N>),
}

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>, const N: usize> AirConstraint<AP>
    for Lookup<AP::Field, E, N>
{
    fn eval(&self, parser: &mut AP) {
        match self {
            Lookup::LogDerivative(log) => log.eval(parser),
        }
    }
}
