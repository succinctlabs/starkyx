use serde::{Deserialize, Serialize};

use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::cubic::element::CubicElement;
use crate::math::prelude::{CubicParameters, *};

/// A log derivative table entry.
///
/// The entry is used to represent a log derivative of the form `multiplier/(beta - value)`. The
/// value for `beta` is implicit and will be provided by the constraints. The `LogEntry` type keeps
/// track of the value of `value` and `multiplier` for a given entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogEntry<T> {
    /// Represents `1 / (beta - value)`.
    Input(T),
    /// Represents `-1 / (beta - value)`.
    Output(T),
    /// Represents `multiplier / (beta - value)`.
    Multiplicity(T, ElementRegister),
}

/// An evaluation of a `LogEntry` instance to be used in constraints.
pub struct LogEntryValue<AP: AirParser> {
    pub value: CubicElement<AP::Var>,
    pub multiplier: AP::Var,
}

impl<T: EvalCubic> LogEntry<T> {
    pub const fn input(value: T) -> Self {
        LogEntry::Input(value)
    }

    pub const fn output(value: T) -> Self {
        LogEntry::Output(value)
    }

    pub const fn multiplicity(value: T, multiplier: ElementRegister) -> Self {
        LogEntry::Multiplicity(value, multiplier)
    }

    #[inline]
    pub fn next(&self) -> Self {
        match self {
            LogEntry::Input(value) => LogEntry::Input(value.next()),
            LogEntry::Output(value) => LogEntry::Output(value.next()),
            LogEntry::Multiplicity(value, multiplier) => {
                LogEntry::Multiplicity(value.next(), multiplier.next())
            }
        }
    }

    pub const fn value(&self) -> &T {
        match self {
            LogEntry::Input(value) => value,
            LogEntry::Output(value) => value,
            LogEntry::Multiplicity(value, _) => value,
        }
    }

    #[inline]
    pub fn eval<AP: CubicParser<E>, E: CubicParameters<AP::Field>>(
        &self,
        parser: &mut AP,
    ) -> LogEntryValue<AP> {
        match self {
            LogEntry::Input(value) => {
                let value = value.eval_cubic(parser);
                let multiplier = parser.one();
                LogEntryValue { value, multiplier }
            }
            LogEntry::Output(value) => {
                let value = value.eval_cubic(parser);
                let multiplier = parser.constant(-AP::Field::ONE);
                LogEntryValue { value, multiplier }
            }
            LogEntry::Multiplicity(value, multiplier) => {
                let value = value.eval_cubic(parser);
                let multiplier = multiplier.eval(parser);
                LogEntryValue { value, multiplier }
            }
        }
    }
}
