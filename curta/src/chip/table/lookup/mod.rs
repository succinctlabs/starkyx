//! Lookup argument
//!

pub mod log_der;

use log_der::LogLookup;

use self::log_der::constraint::LookupConstraint;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::trace::writer::TraceWriter;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
pub enum Lookup<F: Field, E: CubicParameters<F>> {
    Element(LogLookup<ElementRegister, F, E>),
    CubicElement(LogLookup<CubicRegister, F, E>),
}

#[derive(Debug, Clone)]
pub enum LookupChipConstraint<F: Field, E: CubicParameters<F>> {
    Element(LookupConstraint<ElementRegister, F, E>),
    CubicElement(LookupConstraint<CubicRegister, F, E>),
}

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for LookupChipConstraint<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        match self {
            LookupChipConstraint::Element(log) => log.eval(parser),
            LookupChipConstraint::CubicElement(log) => log.eval(parser),
        }
    }
}

impl<F: PrimeField> TraceWriter<F> {
    pub(crate) fn write_lookup<E: CubicParameters<F>>(&self, num_rows: usize, data: &Lookup<F, E>) {
        match data {
            Lookup::Element(log) => self.write_log_lookup(num_rows, log),
            Lookup::CubicElement(log) => self.write_log_lookup(num_rows, log),
        }
    }
}

impl<F: Field, E: CubicParameters<F>> From<LookupConstraint<ElementRegister, F, E>>
    for LookupChipConstraint<F, E>
{
    fn from(constraint: LookupConstraint<ElementRegister, F, E>) -> Self {
        Self::Element(constraint)
    }
}

impl<F: Field, E: CubicParameters<F>> From<LookupConstraint<CubicRegister, F, E>>
    for LookupChipConstraint<F, E>
{
    fn from(constraint: LookupConstraint<CubicRegister, F, E>) -> Self {
        Self::CubicElement(constraint)
    }
}
