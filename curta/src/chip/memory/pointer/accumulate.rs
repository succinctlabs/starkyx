use core::marker::PhantomData;

use plonky2::field::ops::Square;
use serde::{Deserialize, Serialize};

use super::raw::RawPointer;
use crate::air::extension::cubic::CubicParser;
use crate::air::AirConstraint;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::AirParameters;
use crate::math::field::{Field, PrimeField};
use crate::math::prelude::cubic::element::CubicElement;
use crate::math::prelude::CubicParameters;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressedValue<F> {
    Cubic(CubicElement<ArithmeticExpression<F>>),
    Element(ArithmeticExpression<F>),
}

/// Accumulating the pointer value for lookup.
///
/// Given a raw pointer consisting of a challenge `gamma` and a shift, the accumulated value is
/// given by `value + gamma * shift + gamma^2`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointerAccumulator<F, E> {
    /// The raw pointer to be accumulated.
    ptr: RawPointer,
    /// The value in compressed form, consisting of the value and a timestamp.
    value: CompressedValue<F>,
    /// The final digest which is inserted into the bus.
    pub(crate) digest: CubicRegister,
    _marker: PhantomData<E>,
}

impl<F, E> PointerAccumulator<F, E> {
    pub(crate) fn new(ptr: RawPointer, value: CompressedValue<F>, digest: CubicRegister) -> Self {
        Self {
            ptr,
            value,
            digest,
            _marker: PhantomData,
        }
    }

    pub(crate) fn register<L: AirParameters<Field = F, CubicParams = E>>(
        self,
        builder: &mut AirBuilder<L>,
    ) where
        F: PrimeField,
        E: CubicParameters<F>,
    {
        if self.digest.register().is_trace() {
            builder.pointer_row_accumulators.push(self.clone());
            builder.register_constraint(self);
        } else {
            builder.pointer_global_accumulators.push(self.clone());
            builder.register_global_constraint(self)
        }
    }
}

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for PointerAccumulator<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let (powers, shift) = self.ptr.eval(parser);
        let shift = parser.element_from_base_field(shift);

        let value = match self.value.clone() {
            CompressedValue::Element(e) => {
                let value = e.eval(parser)[0];
                parser.element_from_base_field(value)
            }
            CompressedValue::Cubic(e) => CubicElement(e.0.map(|e| e.eval(parser)[0])),
        };

        let mut expected_digest = value;
        let shift_times_challenge = parser.mul_extension(shift, powers[1]);
        // Expected digest is now value + gamma * shift.
        expected_digest = parser.add_extension(expected_digest, shift_times_challenge);
        // Expected digest is now value + gamma * shift + gamma^2.
        expected_digest = parser.add_extension(expected_digest, powers[2]);

        // Compare expected digest with actual digest.
        let digest = self.digest.eval(parser);
        parser.assert_eq_extension(expected_digest, digest);
    }
}

impl<F: Field> TraceWriter<F> {
    pub fn write_ptr_accumulation<E: CubicParameters<F>>(
        &self,
        accumulator: &PointerAccumulator<F, E>,
        row_index: usize,
    ) {
        let ptr_key = accumulator.ptr.read(self, row_index);
        let ptr_challenge = self.read(&ptr_key.challenge, row_index);
        let value = match accumulator.value.clone() {
            CompressedValue::Cubic(cubic) => {
                CubicElement(cubic.0.map(|e| self.read_expression(&e, row_index)[0]))
            }
            CompressedValue::Element(element) => {
                let value = self.read_expression(&element, row_index)[0];
                CubicElement::from_base(value, F::ZERO)
            }
        };

        let digest = value + ptr_challenge * ptr_key.shift + ptr_challenge.square();

        self.write(&accumulator.digest, &digest, row_index);
    }
}
