use plonky2::field::goldilocks_field::GoldilocksField;

use super::gadget::EdScalarMulGadget;
use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::ec::edwards::ed25519::{Ed25519, Ed25519BaseField};
use crate::chip::ec::gadget::EllipticCurveGadget;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::instruction::FpInstruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::evaluation::Digest;
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
    const NUM_FREE_COLUMNS: usize = 67;
    const EXTENDED_COLUMNS: usize = 2282;
    type Instruction = FpInstruction<Ed25519BaseField>;

    fn num_rows_bits() -> usize {
        16
    }
}

pub const ED_NUM_COLUMNS: usize = 1504 + 67 + 2282;

impl<F: PrimeField64, E: CubicParameters<F>> ScalarMulEd25519<F, E> {
    pub fn air() -> (
        Chip<Self>,
        EdScalarMulGadget<F, Ed25519>,
        Vec<ArrayRegister<BitRegister>>,
        Vec<AffinePointRegister<Ed25519>>,
    ) {
        let mut builder = AirBuilder::<Self>::new();

        let res = builder.alloc_unchecked_ec_point();
        let temp = builder.alloc_unchecked_ec_point();
        let scalar_bit = builder.alloc::<BitRegister>();
        let scalar_mul_gadget = builder.ed_scalar_mul::<Ed25519>(&scalar_bit, &res, &temp);

        let scalars = (0..256)
            .into_iter()
            .map(|_| builder.alloc_array_public::<BitRegister>(256))
            .collect::<Vec<_>>();

        let scalars_bits = scalars.iter().map(|s| s.iter()).flatten();

        let points = (0..256)
            .into_iter()
            .map(|_| builder.alloc_public_ec_point())
            .collect::<Vec<_>>();

        let scalar_digest = Digest::from_values(scalars_bits);
        let _ = builder.evaluation(&[scalar_bit], ArithmeticExpression::one(), scalar_digest);

        let point_values = points
            .iter()
            .map(|p| {
                let (x_reg_0, x_reg_1) = p.x.register().get_range();
                let (y_reg_0, y_reg_1) = p.y.register().get_range();
                assert_eq!(x_reg_1, y_reg_0);
                MemorySlice::Public(x_reg_0, y_reg_1 - x_reg_0)
            })
            .collect::<Vec<_>>();
        let point_register = scalar_mul_gadget.temp();

        let point_digest = Digest::from_values(point_values);
        let _inputs_evaluation = builder.evaluation(
            &[point_register.x, point_register.y],
            scalar_mul_gadget.cycle.bit.expr(),
            point_digest,
        );

        let (air, _) = builder.build();

        (air, scalar_mul_gadget, scalars, points)
    }
}
