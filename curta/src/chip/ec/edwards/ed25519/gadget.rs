use curve25519_dalek::edwards::CompressedEdwardsY;
use num::BigUint;

use super::params::Ed25519BaseField;
use super::point::CompressedPointRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::field::register::FieldRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::trace::writer::{AirWriter, TraceWriter};
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::polynomial::to_u16_le_limbs_polynomial;

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

pub trait CompressedPointAirWriter: AirWriter {
    fn write_ec_compressed_point(
        &mut self,
        data: &CompressedPointRegister,
        value: &CompressedEdwardsY,
    ) {
        let mut value_bytes = *value.as_bytes();
        let compressed_sign_bit = Self::Field::from_canonical_u8(value_bytes[31] >> 7);

        //println!("compressed_sign_bit is {:?}", compressed_sign_bit);

        self.write(&data.sign, &compressed_sign_bit);

        // mask the most significant bit
        value_bytes[31] &= 0x7f;

        let y = BigUint::from_bytes_le(&value_bytes);

        let value_y = to_u16_le_limbs_polynomial::<Self::Field, Ed25519BaseField>(&y);
        self.write(&data.y, &value_y);
    }
}

impl<W: AirWriter> CompressedPointAirWriter for W {}

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

        //println!("compressed_sign_bit is {:?}", compressed_sign_bit);

        self.write(&data.sign, &compressed_sign_bit, row_index);

        // mask the most significant bit
        value_bytes[31] &= 0x7f;

        let y = BigUint::from_bytes_le(&value_bytes);

        let value_y = to_u16_le_limbs_polynomial::<F, Ed25519BaseField>(&y);
        self.write(&data.y, &value_y, row_index);
    }
}
