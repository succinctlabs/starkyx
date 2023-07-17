use plonky2::field::goldilocks_field::GoldilocksField;

use super::gadget::EdScalarMulGadget;
use crate::chip::builder::AirBuilder;
use crate::chip::ec::edwards::ed25519::{Ed25519, Ed25519BaseField};
use crate::chip::ec::gadget::EllipticCurveGadget;
use crate::chip::field::instruction::FpInstruction;
use crate::chip::register::bit::BitRegister;
use crate::chip::{AirParameters, Chip};
use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
use crate::math::prelude::*;

pub type EdScalarMulGoldilocks = ScalarMulEd25519<GoldilocksField, GoldilocksCubicParameters>;

#[derive(Debug, Clone)]
pub struct ScalarMulEd25519<F: PrimeField64, E: CubicParameters<F>>(
    core::marker::PhantomData<(F, E)>,
);

impl<F: PrimeField64, E: CubicParameters<F>> const AirParameters for ScalarMulEd25519<F, E> {
    type Field = F;
    type CubicParams = E;

    const NUM_ARITHMETIC_COLUMNS: usize = 1504;
    const NUM_FREE_COLUMNS: usize = 2331;
    type Instruction = FpInstruction<Ed25519BaseField>;

    fn num_rows_bits() -> usize {
        16
    }
}

impl<F: PrimeField64, E: CubicParameters<F>> ScalarMulEd25519<F, E> {
    pub fn air() -> (Chip<Self>, EdScalarMulGadget<F, Ed25519>) {
        let mut builder = AirBuilder::<Self>::new();

        let res = builder.alloc_unchecked_ec_point();
        let temp = builder.alloc_unchecked_ec_point();
        let scalar_bit = builder.alloc::<BitRegister>();
        let scalar_mul_gadget = builder.ed_scalar_mul::<Ed25519>(&scalar_bit, &res, &temp);

        let (air, _) = builder.build();

        (air, scalar_mul_gadget)
    }
}
