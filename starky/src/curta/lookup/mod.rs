//! Lookup table
//!

pub mod log_der;

use log_der::LogLookup;

use super::air::parser::AirParser;

#[derive(Debug, Clone)]
pub enum Lookup {
    LogDerivative(LogLookup),
}

impl Lookup {
    pub fn eval<AP: AirParser>(&self, parser: &mut AP) {
        match self {
            Lookup::LogDerivative(log_lookup) => log_lookup.eval(parser),
        }
    }
}
