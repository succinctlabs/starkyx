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
use crate::plonky2::field::cubic::element::CubicElement;

#[derive(Debug, Clone)]
pub enum Lookup<F: Field, E: CubicParameters<F>> {
    LogDerivative(LogLookup<ElementRegister, F, E>),
    CubicLog(LogLookup<CubicRegister, F, E>),
}

impl<F: Field, E: CubicParameters<F>> Lookup<F, E> {
    pub fn element_table_index(&self) -> Option<fn(F) -> usize> {
        match self {
            Lookup::LogDerivative(log) => log.table_index,
            Lookup::CubicLog(log) => None,
        }
    }

    pub fn cubic_table_index(&self) -> Option<fn(CubicElement<F>) -> usize> {
        match self {
            Lookup::LogDerivative(log) => None,
            Lookup::CubicLog(log) => log.table_index,
        }
    }
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
