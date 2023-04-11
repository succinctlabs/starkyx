use super::add::{EcAddData, FromEdwardsAdd};
use super::*;
use crate::arithmetic::builder::ChipBuilder;
use crate::arithmetic::chip::ChipParameters;
use crate::arithmetic::register::BitRegister;

#[derive(Clone, Copy)]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct EcScalarMulData<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize> {
    scalar_bit: BitRegister,
    add: EcAddData<E, N_LIMBS>,
    double: EcAddData<E, N_LIMBS>,
}

impl<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize> EcAddData<E, N_LIMBS> {
    pub const fn num_ed_scalar_mul_columns() -> usize {
        1 + 2 * EcAddData::<E, N_LIMBS>::num_ed_add_columns()
    }
}

// impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
//     #[allow(non_snake_case)]
//     pub fn ed_scalar_mul<E: EdwardsParameters<N>, const N: usize>(
//         &mut self,
//         P: &AffinePointRegister<E, N>,
//         bit : &BitRegister,
//         result: &AffinePointRegister<E, N>,
//     ) -> Result<()>//EcScalarMulData<E, N>>
//     where
//         L::Instruction: FromEdwardsAdd<E, N>, {
//             let result = self.alloc_ec_point()?;

//             Ok(())
//         }
// }
