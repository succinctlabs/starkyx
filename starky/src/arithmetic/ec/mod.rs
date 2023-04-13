pub mod edwards;

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::register::FieldRegister;
use crate::arithmetic::builder::ChipBuilder;
use crate::arithmetic::chip::ChipParameters;
use crate::arithmetic::field::FieldParameters;
use crate::arithmetic::register::Register;
use crate::arithmetic::trace::TraceHandle;

pub const LIMB: u32 = 2u32.pow(16);

pub trait EllipticCurveParameters: Send + Sync + Copy + 'static {
    type FieldParam: FieldParameters;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinePoint<E: EllipticCurveParameters> {
    pub x: BigUint,
    pub y: BigUint,
    _marker: std::marker::PhantomData<E>,
}

impl<E: EllipticCurveParameters> AffinePoint<E> {
    #[allow(dead_code)]
    fn new(x: BigUint, y: BigUint) -> Self {
        Self {
            x,
            y,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct AffinePointRegister<E: EllipticCurveParameters> {
    x: FieldRegister<E::FieldParam>,
    y: FieldRegister<E::FieldParam>,
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
    /// Allocates registers for an affine elliptic curve point.
    ///
    /// The entries are range-checked to be less than 2^16.

    /// Allocates registers for a next affine elliptic curve point without range-checking.
    pub fn alloc_unchecked_ec_point<E: EllipticCurveParameters>(
        &mut self,
    ) -> Result<AffinePointRegister<E>> {
        let x = self.alloc_local::<FieldRegister<E::FieldParam>>().unwrap();
        let y = self.alloc_local::<FieldRegister<E::FieldParam>>().unwrap();
        Ok(AffinePointRegister::<E>::from_field_registers(x, y))
    }

    pub fn alloc_ec_point<E: EllipticCurveParameters>(&mut self) -> Result<AffinePointRegister<E>> {
        self.alloc_local_ec_point()
    }

    pub fn alloc_local_ec_point<E: EllipticCurveParameters>(
        &mut self,
    ) -> Result<AffinePointRegister<E>> {
        let x = self.alloc_local::<FieldRegister<E::FieldParam>>()?;
        let y = self.alloc_local::<FieldRegister<E::FieldParam>>()?;
        Ok(AffinePointRegister { x, y })
    }

    pub fn alloc_next_ec_point<E: EllipticCurveParameters>(
        &mut self,
    ) -> Result<AffinePointRegister<E>> {
        let x = self.alloc_next::<FieldRegister<E::FieldParam>>()?;
        let y = self.alloc_next::<FieldRegister<E::FieldParam>>()?;
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

impl<F: RichField + Extendable<D>, const D: usize> TraceHandle<F, D> {
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
        x: FieldRegister<E::FieldParam>,
        y: FieldRegister<E::FieldParam>,
    ) -> Self {
        Self { x, y }
    }
}
