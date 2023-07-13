use super::point::{AffinePoint, AffinePointRegister};
use super::EllipticCurveParameters;
use crate::chip::builder::AirBuilder;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::field::register::FieldRegister;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::to_u16_le_limbs_polynomial;

pub trait EllipticCurveGadget<E: EllipticCurveParameters> {
    fn alloc_unchecked_ec_point(&mut self) -> AffinePointRegister<E>;

    fn alloc_local_ec_point(&mut self) -> AffinePointRegister<E>;

    fn alloc_next_ec_point(&mut self) -> AffinePointRegister<E>;

    fn alloc_ec_point(&mut self) -> AffinePointRegister<E> {
        self.alloc_local_ec_point()
    }
}

pub trait EllipticCurveWriter<E: EllipticCurveParameters> {
    fn write_ec_point(
        &self,
        data: &AffinePointRegister<E>,
        value: &AffinePoint<E>,
        row_index: usize,
    );
}

impl<L: AirParameters, E: EllipticCurveParameters> EllipticCurveGadget<E> for AirBuilder<L> {
    /// Allocates registers for a next affine elliptic curve point without range-checking.
    fn alloc_unchecked_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = FieldRegister::<E::BaseField>::from_register(
            self.get_local_memory(E::BaseField::NB_LIMBS),
        );
        let y = FieldRegister::<E::BaseField>::from_register(
            self.get_local_memory(E::BaseField::NB_LIMBS),
        );
        AffinePointRegister::new(x, y)
    }

    fn alloc_local_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = self.alloc::<FieldRegister<E::BaseField>>();
        let y = self.alloc::<FieldRegister<E::BaseField>>();
        AffinePointRegister::new(x, y)
    }

    fn alloc_next_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = self.alloc_next::<FieldRegister<E::BaseField>>();
        let y = self.alloc_next::<FieldRegister<E::BaseField>>();
        AffinePointRegister::new(x, y)
    }
}

impl<F: Field, E: EllipticCurveParameters> EllipticCurveWriter<E> for TraceWriter<F> {
    fn write_ec_point(
        &self,
        data: &AffinePointRegister<E>,
        value: &AffinePoint<E>,
        row_index: usize,
    ) {
        let value_x = to_u16_le_limbs_polynomial::<F, E::BaseField>(&value.x);
        let value_y = to_u16_le_limbs_polynomial::<F, E::BaseField>(&value.y);
        self.write_value(&data.x, &value_x, row_index);
        self.write_value(&data.y, &value_y, row_index);
    }
}
