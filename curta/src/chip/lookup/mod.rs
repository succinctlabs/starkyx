//! Lookup table
//!

pub mod log_der;

use log_der::LogLookup;

use crate::air::extension::ExtensionParser;
use crate::air::AirConstraint;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum Lookup<F: Field, E: ExtensionField<F>, const N: usize>
where
    [(); E::D]:,
{
    LogDerivative(LogLookup<F, E, N>),
}

impl<E: ExtensionField<AP::Field>, AP: ExtensionParser<E>, const N: usize> AirConstraint<AP>
    for Lookup<AP::Field, E, N>
where
    [(); E::D]:,
{
    fn eval(&self, parser: &mut AP) {
        match self {
            Lookup::LogDerivative(log) => log.eval(parser),
        }
    }
}
