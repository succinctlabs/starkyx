use curve25519_dalek::edwards::CompressedEdwardsY;
use num::BigUint;

use super::edwards::ed25519::Ed25519BaseField;
use super::point::{AffinePoint, AffinePointRegister, CompressedPointRegister};
use super::EllipticCurve;
use crate::chip::builder::AirBuilder;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::field::register::FieldRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::utils::field_limbs_to_biguint;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::to_u16_le_limbs_polynomial;

pub trait EllipticCurveGadget<E: EllipticCurve> {
    fn alloc_unchecked_ec_point(&mut self) -> AffinePointRegister<E>;

    fn alloc_local_ec_point(&mut self) -> AffinePointRegister<E>;

    fn alloc_ec_point(&mut self) -> AffinePointRegister<E> {
        self.alloc_local_ec_point()
    }

    fn alloc_global_ec_point(&mut self) -> AffinePointRegister<E>;

    fn alloc_public_ec_point(&mut self) -> AffinePointRegister<E>;
}

pub trait EllipticCurveWriter<E: EllipticCurve> {
    fn read_ec_point(&self, data: &AffinePointRegister<E>, row_index: usize) -> AffinePoint<E>;

    fn write_ec_point(
        &self,
        data: &AffinePointRegister<E>,
        value: &AffinePoint<E>,
        row_index: usize,
    );
}

impl<L: AirParameters, E: EllipticCurve> EllipticCurveGadget<E> for AirBuilder<L> {
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

    fn alloc_global_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = self.alloc_global::<FieldRegister<E::BaseField>>();
        let y = self.alloc_global::<FieldRegister<E::BaseField>>();
        AffinePointRegister::new(x, y)
    }

    fn alloc_public_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = self.alloc_public::<FieldRegister<E::BaseField>>();
        let y = self.alloc_public::<FieldRegister<E::BaseField>>();
        AffinePointRegister::new(x, y)
    }
}

impl<F: PrimeField64, E: EllipticCurve> EllipticCurveWriter<E> for TraceWriter<F> {
    fn read_ec_point(&self, data: &AffinePointRegister<E>, row_index: usize) -> AffinePoint<E> {
        let p_x = self.read(&data.x, row_index);
        let p_y = self.read(&data.y, row_index);

        let x = field_limbs_to_biguint(p_x.coefficients());
        let y = field_limbs_to_biguint(p_y.coefficients());

        AffinePoint::<E>::new(x, y)
    }

    fn write_ec_point(
        &self,
        data: &AffinePointRegister<E>,
        value: &AffinePoint<E>,
        row_index: usize,
    ) {
        let value_x = to_u16_le_limbs_polynomial::<F, E::BaseField>(&value.x);
        let value_y = to_u16_le_limbs_polynomial::<F, E::BaseField>(&value.y);
        self.write(&data.x, &value_x, row_index);
        self.write(&data.y, &value_y, row_index);
    }
}

pub trait CompressedPointGadget {
    fn alloc_local_ec_compressed_point(&mut self) -> CompressedPointRegister;

    fn alloc_ec_compressed_point(&mut self) -> CompressedPointRegister {
        self.alloc_local_ec_compressed_point()
    }

    fn alloc_global_ec_compressed_point(&mut self) -> CompressedPointRegister;

    fn alloc_public_ec_compressed_point(&mut self) -> CompressedPointRegister;
}

pub trait CompressedPointWriter {
    fn write_ec_compressed_point(
        &self,
        data: &CompressedPointRegister,
        value: &CompressedEdwardsY,
        row_index: usize,
    );
}

impl<L: AirParameters> CompressedPointGadget for AirBuilder<L> {
    fn alloc_local_ec_compressed_point(&mut self) -> CompressedPointRegister {
        let y = self.alloc::<FieldRegister<Ed25519BaseField>>();
        let sign = self.alloc::<BitRegister>();
        CompressedPointRegister::new(sign, y)
    }

    fn alloc_global_ec_compressed_point(&mut self) -> CompressedPointRegister {
        let y = self.alloc_global::<FieldRegister<Ed25519BaseField>>();
        let sign = self.alloc_global::<BitRegister>();
        CompressedPointRegister::new(sign, y)
    }

    fn alloc_public_ec_compressed_point(&mut self) -> CompressedPointRegister {
        let y = self.alloc_public::<FieldRegister<Ed25519BaseField>>();
        let sign = self.alloc_public::<BitRegister>();
        CompressedPointRegister::new(sign, y)
    }
}

impl<F: PrimeField64> CompressedPointWriter for TraceWriter<F> {
    fn write_ec_compressed_point(
        &self,
        data: &CompressedPointRegister,
        value: &CompressedEdwardsY,
        row_index: usize,
    ) {
        let mut value_bytes = *value.as_bytes();
        let compressed_sign_bit = F::from_canonical_u8(value_bytes[31] >> 7);

        self.write(&data.sign, &compressed_sign_bit, row_index);

        // mask the most significant bit
        value_bytes[31] &= 0x7f;

        let y = BigUint::from_bytes_le(&value_bytes);

        let value_y = to_u16_le_limbs_polynomial::<F, Ed25519BaseField>(&y);
        self.write(&data.y, &value_y, row_index);
    }
}
