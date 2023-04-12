pub mod edwards;

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::register::DataRegister;
use crate::arithmetic::builder::ChipBuilder;
use crate::arithmetic::chip::ChipParameters;
use crate::arithmetic::field::{FieldParameters, FieldRegister};
use crate::arithmetic::trace::TraceHandle;

pub const LIMB: u32 = 2u32.pow(16);

pub trait EllipticCurveParameters<const N_LIMBS: usize>: Send + Sync + Copy + 'static {
    type FieldParam: FieldParameters<N_LIMBS>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinePoint<E: EllipticCurveParameters<N>, const N: usize> {
    pub x: BigUint,
    pub y: BigUint,
    _marker: std::marker::PhantomData<E>,
}

impl<E: EllipticCurveParameters<N>, const N: usize> AffinePoint<E, N> {
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
pub struct AffinePointRegister<E: EllipticCurveParameters<N_LIMBS>, const N_LIMBS: usize> {
    x: FieldRegister<E::FieldParam, N_LIMBS>,
    y: FieldRegister<E::FieldParam, N_LIMBS>,
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
    /// Allocates registers for an affine elliptic curve point.
    ///
    /// The entries are range-checked to be less than 2^16.
    pub fn alloc_ec_point<E: EllipticCurveParameters<N_LIMBS>, const N_LIMBS: usize>(
        &mut self,
    ) -> Result<AffinePointRegister<E, N_LIMBS>> {
        self.alloc_local_ec_point()
    }

    /// Allocates registers for a next affine elliptic curve point without range-checking.
    pub fn alloc_unchecked_ec_point<E: EllipticCurveParameters<N_LIMBS>, const N_LIMBS: usize>(
        &mut self,
    ) -> Result<AffinePointRegister<E, N_LIMBS>> {
        let x = FieldRegister::<E::FieldParam, N_LIMBS>::from_raw_register(
            self.get_local_memory(N_LIMBS).unwrap(),
        );
        let y = FieldRegister::<E::FieldParam, N_LIMBS>::from_raw_register(
            self.get_local_memory(N_LIMBS).unwrap(),
        );
        Ok(AffinePointRegister::<E, N_LIMBS>::from_field_registers(
            x, y,
        ))
    }

    pub fn alloc_local_ec_point<E: EllipticCurveParameters<N_LIMBS>, const N_LIMBS: usize>(
        &mut self,
    ) -> Result<AffinePointRegister<E, N_LIMBS>> {
        let x = self.alloc_local::<FieldRegister<E::FieldParam, N_LIMBS>>()?;
        let y = self.alloc_local::<FieldRegister<E::FieldParam, N_LIMBS>>()?;
        Ok(AffinePointRegister { x, y })
    }

    pub fn alloc_next_ec_point<E: EllipticCurveParameters<N_LIMBS>, const N_LIMBS: usize>(
        &mut self,
    ) -> Result<AffinePointRegister<E, N_LIMBS>> {
        let x = self.alloc_next::<FieldRegister<E::FieldParam, N_LIMBS>>()?;
        let y = self.alloc_next::<FieldRegister<E::FieldParam, N_LIMBS>>()?;
        Ok(AffinePointRegister { x, y })
    }

    pub fn write_ec_point<E: EllipticCurveParameters<N_LIMBS>, const N_LIMBS: usize>(
        &mut self,
        data: &AffinePointRegister<E, N_LIMBS>,
    ) -> Result<()> {
        self.write_data(&data.x)?;
        self.write_data(&data.y)?;
        Ok(())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceHandle<F, D> {
    #[allow(dead_code)]
    fn write_ec_point<E: EllipticCurveParameters<N_LIMBS>, const N_LIMBS: usize>(
        &self,
        row_index: usize,
        point: &AffinePoint<E, N_LIMBS>,
        data: &AffinePointRegister<E, N_LIMBS>,
    ) -> Result<()> {
        self.write_field(row_index, &point.x, data.x)?;
        self.write_field(row_index, &point.y, data.y)
    }
}

impl<E: EllipticCurveParameters<N_LIMBS>, const N_LIMBS: usize> AffinePointRegister<E, N_LIMBS> {
    pub fn next(&self) -> Self {
        Self {
            x: self.x.next(),
            y: self.y.next(),
        }
    }

    pub fn from_field_registers(
        x: FieldRegister<E::FieldParam, N_LIMBS>,
        y: FieldRegister<E::FieldParam, N_LIMBS>,
    ) -> Self {
        Self { x, y }
    }
}
