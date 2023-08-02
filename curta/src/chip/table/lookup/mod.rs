//! Lookup argument
//!

pub mod log_der;

use log_der::LogLookup;

use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::TraceWriter;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum Lookup<F: Field, E: CubicParameters<F>> {
    LogDerivative(LogLookup<ElementRegister, F, E>),
    CubicLog(LogLookup<CubicRegister, F, E>),
}

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP> for Lookup<AP::Field, E> {
    fn eval(&self, parser: &mut AP) {
        match self {
            Lookup::LogDerivative(log) => log.eval(parser),
            Lookup::CubicLog(log) => log.eval(parser),
        }
    }
}
impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_lookup<E: CubicParameters<F>>(&self, num_rows: usize, data: &Lookup<F, E>) {
        match data {
            Lookup::LogDerivative(log) => self.write_log_lookup(num_rows, log),
            Lookup::CubicLog(log) => self.write_log_lookup(num_rows, log),
        }
    }
}
