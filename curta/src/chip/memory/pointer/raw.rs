use serde::{Deserialize, Serialize};

use super::accumulate::{CompressedValue, PointerAccumulator};
use super::key::RawPointerKey;
use crate::air::extension::cubic::CubicParser;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::AirParameters;
use crate::math::field::Field;
use crate::math::prelude::cubic::element::CubicElement;
use crate::math::prelude::CubicParameters;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RawPointer {
    challenge: CubicRegister,
    element_shift: Option<ElementRegister>,
    constant_shift: Option<i32>,
}

impl RawPointer {
    pub(crate) fn new(
        challenge: CubicRegister,
        element_shift: Option<ElementRegister>,
        constant_shift: Option<i32>,
    ) -> Self {
        Self {
            challenge,
            element_shift,
            constant_shift,
        }
    }

    pub(crate) fn from_challenge(challenge: CubicRegister) -> Self {
        Self {
            challenge,
            element_shift: None,
            constant_shift: None,
        }
    }

    pub fn is_trace(&self) -> bool {
        self.element_shift.map(|e| e.is_trace()).unwrap_or(false)
    }

    pub fn accumulate<L: AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        value: ArithmeticExpression<L::Field>,
    ) -> CubicRegister {
        let digest = if value.is_trace() || self.is_trace() {
            builder.alloc_extended()
        } else {
            builder.alloc_global()
        };

        let value = CompressedValue::Element(value);
        let accumulator = PointerAccumulator::new(*self, value, digest);
        accumulator.register(builder);

        digest
    }

    pub fn accumulate_cubic<L: AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        value: CubicElement<ArithmeticExpression<L::Field>>,
    ) -> CubicRegister {
        let digest = if !value.as_slice().iter().all(|e| !e.is_trace()) || self.is_trace() {
            builder.alloc_extended()
        } else {
            builder.alloc_global()
        };
        let value = CompressedValue::Cubic(value);
        let accumulator = PointerAccumulator::new(*self, value, digest);
        accumulator.register(builder);

        digest
    }

    pub fn eval<E: CubicParameters<AP::Field>, AP: CubicParser<E>>(
        &self,
        parser: &mut AP,
    ) -> CubicElement<AP::Var> {
        let challenge = self.challenge.eval(parser);

        let shift = match (self.element_shift, self.constant_shift) {
            (Some(e), None) => Some(e.eval(parser)),
            (None, Some(c)) => Some(parser.constant(i32_to_field(c))),
            (Some(e), Some(c)) => {
                let element = e.eval(parser);
                let constant = i32_to_field(c);
                Some(parser.add_const(element, constant))
            }
            (None, None) => None,
        };

        let shift_ext = shift.map(|e| parser.element_from_base_field(e));
        shift_ext
            .map(|e| parser.add_extension(challenge, e))
            .unwrap_or(challenge)
    }

    pub fn read<F: Field>(&self, writer: &TraceWriter<F>, row_index: usize) -> RawPointerKey<F> {
        let element_shift = self
            .element_shift
            .map(|s| writer.read(&s, row_index))
            .unwrap_or(F::ZERO);
        let constant_shift = self.constant_shift.map(i32_to_field).unwrap_or(F::ZERO);
        let shift = element_shift + constant_shift;
        RawPointerKey::new(self.challenge, shift)
    }

    pub fn read_from_air<F: Field>(&self, writer: &impl AirWriter<Field = F>) -> RawPointerKey<F> {
        let element_shift = self
            .element_shift
            .map(|s| writer.read(&s))
            .unwrap_or(F::ZERO);
        let constant_shift = self.constant_shift.map(i32_to_field).unwrap_or(F::ZERO);
        let shift = element_shift + constant_shift;
        RawPointerKey::new(self.challenge, shift)
    }
}

fn i32_to_field<F: Field>(x: i32) -> F {
    if x < 0 {
        -F::from_canonical_u32(-x as u32)
    } else {
        F::from_canonical_u32(x as u32)
    }
}
