use core::marker::PhantomData;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointerAccumulator<F, E> {
    ptr: RawPointer,
    value: CompressedValue<F>,
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
        let challenge = self.ptr.eval(parser);
        let digest = self.digest.eval(parser);

        let expected_value = match self.value.clone() {
            CompressedValue::Cubic(cubic) => {
                let value = CubicElement(cubic.0.map(|e| e.eval(parser)[0]));
                parser.mul_extension(challenge, value)
            }
            CompressedValue::Element(element) => {
                let value_base = element.eval(parser)[0];
                let value = parser.element_from_base_field(value_base);
                parser.mul_extension(challenge, value)
            }
        };

        parser.assert_eq_extension(expected_value, digest);
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
        let ptr_value = ptr_challenge + CubicElement::from_base(ptr_key.shift, F::ZERO);

        let value = match accumulator.value.clone() {
            CompressedValue::Cubic(cubic) => {
                let value = CubicElement(cubic.0.map(|e| self.read_expression(&e, row_index)[0]));
                value * ptr_value
            }
            CompressedValue::Element(element) => {
                let value = self.read_expression(&element, row_index)[0];
                ptr_value * CubicElement::from_base(value, F::ZERO)
            }
        };

        self.write(&accumulator.digest, &value, row_index);
    }
}
