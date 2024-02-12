use serde::{Deserialize, Serialize};

use super::accumulate::{CompressedValue, PointerAccumulator};
use super::key::RawPointerKey;
use crate::air::extension::cubic::CubicParser;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
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
    /// The powers `1, gamma, gamma^2, ...` of the challenge identifying the unique pointer.
    powers: ArrayRegister<CubicRegister>,
    element_shift: Option<ElementRegister>,
    constant_shift: Option<i32>,
}

impl RawPointer {
    pub(crate) fn new(
        powers: ArrayRegister<CubicRegister>,
        element_shift: Option<ElementRegister>,
        constant_shift: Option<i32>,
    ) -> Self {
        Self {
            powers,
            element_shift,
            constant_shift,
        }
    }

    pub(crate) fn from_challenge(powers: ArrayRegister<CubicRegister>) -> Self {
        Self {
            powers,
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
    ) -> ([CubicElement<AP::Var>; 3], AP::Var) {
        let challenges = self.powers.eval_array::<_, 3>(parser);

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

        (challenges, shift.unwrap_or(parser.zero()))
    }

    pub fn shift_expr<F: Field>(&self) -> ArithmeticExpression<F> {
        match (self.element_shift, self.constant_shift) {
            (Some(e), None) => e.expr(),
            (None, Some(c)) => ArithmeticExpression::from_constant(i32_to_field(c)),
            (Some(e), Some(c)) => {
                let element = e.expr::<F>();
                let constant = i32_to_field::<F>(c);
                element + constant
            }
            (None, None) => ArithmeticExpression::zero(),
        }
    }

    pub fn read<F: Field>(&self, writer: &TraceWriter<F>, row_index: usize) -> RawPointerKey<F> {
        let element_shift = self
            .element_shift
            .map(|s| writer.read(&s, row_index))
            .unwrap_or(F::ZERO);
        let constant_shift = self.constant_shift.map(i32_to_field).unwrap_or(F::ZERO);
        let shift = element_shift + constant_shift;
        RawPointerKey::new(self.powers.get(1), shift)
    }

    pub fn read_from_air<F: Field>(&self, writer: &impl AirWriter<Field = F>) -> RawPointerKey<F> {
        let element_shift = self
            .element_shift
            .map(|s| writer.read(&s))
            .unwrap_or(F::ZERO);
        let constant_shift = self.constant_shift.map(i32_to_field).unwrap_or(F::ZERO);
        let shift = element_shift + constant_shift;
        RawPointerKey::new(self.powers.get(1), shift)
    }
}

fn i32_to_field<F: Field>(x: i32) -> F {
    if x < 0 {
        -F::from_canonical_u32(-x as u32)
    } else {
        F::from_canonical_u32(x as u32)
    }
}
