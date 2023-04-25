pub mod affine;
pub mod edwards;

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use self::affine::AffinePoint;
use super::parameters::EllipticCurveParameters;
use super::register::{FieldRegister, RegisterSerializable};
use crate::arithmetic::builder::StarkBuilder;
use crate::arithmetic::chip::StarkParameters;
use crate::arithmetic::parameters::FieldParameters;
use crate::arithmetic::register::Register;
use crate::arithmetic::trace::TraceWriter;

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct AffinePointRegister<E: EllipticCurveParameters> {
    x: FieldRegister<E::FieldParameters>,
    y: FieldRegister<E::FieldParameters>,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    /// Allocates registers for an affine elliptic curve point.
    ///
    /// The entries are range-checked to be less than 2^16.

    /// Allocates registers for a next affine elliptic curve point without range-checking.
    pub fn alloc_unchecked_ec_point<E: EllipticCurveParameters>(
        &mut self,
    ) -> Result<AffinePointRegister<E>> {
        let x = FieldRegister::<E::FieldParameters>::from_register(
            self.get_local_memory(E::FieldParameters::NB_LIMBS),
        );
        let y = FieldRegister::<E::FieldParameters>::from_register(
            self.get_local_memory(E::FieldParameters::NB_LIMBS),
        );
        Ok(AffinePointRegister::<E>::from_field_registers(x, y))
    }

    pub fn alloc_ec_point<E: EllipticCurveParameters>(&mut self) -> Result<AffinePointRegister<E>> {
        self.alloc_local_ec_point()
    }

    pub fn alloc_local_ec_point<E: EllipticCurveParameters>(
        &mut self,
    ) -> Result<AffinePointRegister<E>> {
        let x = self.alloc::<FieldRegister<E::FieldParameters>>();
        let y = self.alloc::<FieldRegister<E::FieldParameters>>();
        Ok(AffinePointRegister { x, y })
    }

    pub fn alloc_next_ec_point<E: EllipticCurveParameters>(
        &mut self,
    ) -> Result<AffinePointRegister<E>> {
        let x = self.alloc_next::<FieldRegister<E::FieldParameters>>()?;
        let y = self.alloc_next::<FieldRegister<E::FieldParameters>>()?;
        Ok(AffinePointRegister { x, y })
    }

    pub fn write_ec_point<E: EllipticCurveParameters>(
        &mut self,
        data: &AffinePointRegister<E>,
    ) -> Result<()> {
        self.write_data(&data.x)?;
        self.write_data(&data.y)?;
        Ok(())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    #[allow(dead_code)]
    fn write_ec_point<E: EllipticCurveParameters>(
        &self,
        row_index: usize,
        point: &AffinePoint<E>,
        data: &AffinePointRegister<E>,
    ) -> Result<()> {
        self.write_field(row_index, &point.x, data.x)?;
        self.write_field(row_index, &point.y, data.y)
    }
}

impl<E: EllipticCurveParameters> AffinePointRegister<E> {
    pub fn next(&self) -> Self {
        Self {
            x: self.x.next(),
            y: self.y.next(),
        }
    }

    pub fn from_field_registers(
        x: FieldRegister<E::FieldParameters>,
        y: FieldRegister<E::FieldParameters>,
    ) -> Self {
        Self { x, y }
    }
}
