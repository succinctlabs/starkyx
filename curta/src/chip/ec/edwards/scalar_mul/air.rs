use plonky2::field::goldilocks_field::GoldilocksField;

use super::gadget::EdScalarMulGadget;
use crate::chip::builder::AirBuilder;
use crate::chip::ec::edwards::ed25519::{Ed25519, Ed25519BaseField};
use crate::chip::ec::gadget::EllipticCurveGadget;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::instruction::FpInstruction;
use crate::chip::instruction::set::AirInstruction;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::evaluation::{BitEvaluation, Digest};
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
    const NUM_FREE_COLUMNS: usize = 75;
    const EXTENDED_COLUMNS: usize = 2292;
    type Instruction = FpInstruction<Ed25519BaseField>;

    fn num_rows_bits() -> usize {
        16
    }
}

pub const ED_NUM_COLUMNS: usize = 1504 + 75 + 2292;

impl<F: PrimeField64, E: CubicParameters<F>> ScalarMulEd25519<F, E> {
    #[allow(clippy::type_complexity)]
    pub fn air() -> (
        Chip<Self>,
        EdScalarMulGadget<F, Ed25519>,
        Vec<ArrayRegister<ElementRegister>>,
        Vec<AffinePointRegister<Ed25519>>,
        Vec<AffinePointRegister<Ed25519>>,
        (
            BitEvaluation<F>,
            AirInstruction<F, FpInstruction<Ed25519BaseField>>,
            AirInstruction<F, FpInstruction<Ed25519BaseField>>,
        ),
    ) {
        let mut builder = AirBuilder::<Self>::new_with_public_inputs(18432);

        let res = builder.alloc_unchecked_ec_point();
        let temp = builder.alloc_unchecked_ec_point();
        let scalar_bit = builder.alloc::<BitRegister>();
        let scalar_mul_gadget = builder.ed_scalar_mul::<Ed25519>(&scalar_bit, &res, &temp);

        let scalars_limbs = (0..256)
            .map(|_| builder.alloc_array_public::<ElementRegister>(8))
            .collect::<Vec<_>>();

        let scalars_u32 = scalars_limbs.iter().flat_map(|s| s.iter());

        let input_points = (0..256)
            .map(|_| builder.alloc_public_ec_point())
            .collect::<Vec<_>>();

        let output_points = (0..256)
            .map(|_| builder.alloc_public_ec_point())
            .collect::<Vec<_>>();

        let scalar_digest = Digest::from_values(scalars_u32);
        // let (bit_eval, write_first, set_bit) = builder.bit_evaluation(&[scalar_bit], ArithmeticExpression::one(), scalar_digest);
        let (bit_eval, set_last, set_bit) = builder.bit_evaluation(&scalar_bit, scalar_digest);

        let input_point_values = input_points
            .iter()
            .map(|p| {
                let (x_reg_0, x_reg_1) = p.x.register().get_range();
                let (y_reg_0, y_reg_1) = p.y.register().get_range();
                assert_eq!(x_reg_1, y_reg_0);
                MemorySlice::Global(x_reg_0, y_reg_1 - x_reg_0)
            })
            .collect::<Vec<_>>();
        let input_point_register = scalar_mul_gadget.temp();

        let input_point_digest = Digest::from_values(input_point_values);
        builder.evaluation(
            &[input_point_register.x, input_point_register.y],
            scalar_mul_gadget.cycle.start_bit.expr(),
            input_point_digest,
        );

        let output_point_values = output_points
            .iter()
            .map(|p| {
                let (x_reg_0, x_reg_1) = p.x.register().get_range();
                let (y_reg_0, y_reg_1) = p.y.register().get_range();
                assert_eq!(x_reg_1, y_reg_0);
                MemorySlice::Global(x_reg_0, y_reg_1 - x_reg_0)
            })
            .collect::<Vec<_>>();
        let output_point_register = scalar_mul_gadget.result();

        let output_point_digest = Digest::from_values(output_point_values);
        builder.evaluation(
            &[output_point_register.x, output_point_register.y],
            scalar_mul_gadget.cycle.end_bit.expr(),
            output_point_digest,
        );

        let air = builder.build();

        (
            air,
            scalar_mul_gadget,
            scalars_limbs,
            input_points,
            output_points,
            (bit_eval, set_last, set_bit),
        )
    }
}
