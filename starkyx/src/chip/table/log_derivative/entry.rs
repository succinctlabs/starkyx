use serde::{Deserialize, Serialize};

use crate::air::extension::cubic::CubicParser;
use crate::chip::register::cubic::EvalCubic;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::prelude::cubic::element::CubicElement;
use crate::math::prelude::cubic::extension::CubicExtension;
use crate::math::prelude::*;

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
    InputMultiplicity(T, ElementRegister),
    // Represents `-multiplier / (beta - value)`
    OutputMultiplicity(T, ElementRegister),
}

/// An evaluation of a `LogEntry` instance to be used in constraints.
pub struct LogEntryValue<V> {
    pub value: CubicElement<V>,
    pub multiplier: V,
}

impl<F: Field> LogEntryValue<F> {
    #[inline]
    pub fn evaluate<E: CubicParameters<F>>(
        &self,
        beta: CubicExtension<F, E>,
    ) -> CubicExtension<F, E> {
        CubicExtension::from_base_field(self.multiplier) / (beta - CubicExtension::from(self.value))
    }
}

impl<T: EvalCubic> LogEntry<T> {
    pub const fn input(value: T) -> Self {
        LogEntry::Input(value)
    }

    pub const fn output(value: T) -> Self {
        LogEntry::Output(value)
    }

    pub const fn input_with_multiplicity(value: T, multiplier: ElementRegister) -> Self {
        LogEntry::InputMultiplicity(value, multiplier)
    }

    pub const fn output_with_multiplicity(value: T, multiplier: ElementRegister) -> Self {
        LogEntry::OutputMultiplicity(value, multiplier)
    }

    #[inline]
    pub fn next(&self) -> Self {
        match self {
            LogEntry::Input(value) => LogEntry::Input(value.next()),
            LogEntry::Output(value) => LogEntry::Output(value.next()),
            LogEntry::InputMultiplicity(value, multiplier) => {
                LogEntry::InputMultiplicity(value.next(), multiplier.next())
            }
            LogEntry::OutputMultiplicity(value, multiplier) => {
                LogEntry::OutputMultiplicity(value.next(), multiplier.next())
            }
        }
    }

    pub const fn value(&self) -> &T {
        match self {
            LogEntry::Input(value) => value,
            LogEntry::Output(value) => value,
            LogEntry::InputMultiplicity(value, _) => value,
            LogEntry::OutputMultiplicity(value, _) => value,
        }
    }

    #[inline]
    pub fn eval<AP: CubicParser<E>, E: CubicParameters<AP::Field>>(
        &self,
        parser: &mut AP,
    ) -> LogEntryValue<AP::Var> {
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
            LogEntry::InputMultiplicity(value, multiplier) => {
                let value = value.eval_cubic(parser);
                let multiplier = multiplier.eval(parser);
                LogEntryValue { value, multiplier }
            }
            LogEntry::OutputMultiplicity(value, multiplier) => {
                let value = value.eval_cubic(parser);
                let mut multiplier = multiplier.eval(parser);
                multiplier = parser.neg(multiplier);
                LogEntryValue { value, multiplier }
            }
        }
    }

    #[inline]
    pub fn read_from_slice<F: Field>(&self, slice: &[F]) -> LogEntryValue<F> {
        match self {
            LogEntry::Input(value) => {
                let value = T::trace_value_as_cubic(value.read_from_slice(slice));
                LogEntryValue {
                    value,
                    multiplier: F::ONE,
                }
            }
            LogEntry::Output(value) => {
                let value = T::trace_value_as_cubic(value.read_from_slice(slice));
                let multiplier = -F::ONE;
                LogEntryValue { value, multiplier }
            }
            LogEntry::InputMultiplicity(value, multiplier) => {
                let value = T::trace_value_as_cubic(value.read_from_slice(slice));
                let multiplier = multiplier.read_from_slice(slice);
                LogEntryValue { value, multiplier }
            }
            LogEntry::OutputMultiplicity(value, multiplier) => {
                let value = T::trace_value_as_cubic(value.read_from_slice(slice));
                let mut multiplier = multiplier.read_from_slice(slice);
                multiplier = -multiplier;
                LogEntryValue { value, multiplier }
            }
        }
    }
}
