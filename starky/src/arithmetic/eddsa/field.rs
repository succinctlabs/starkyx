use super::*;
use crate::arithmetic::circuit::EmulatedCircuitLayout;
use crate::arithmetic::arithmetic_stark;
use crate::arithmetic::polynomial::Polynomial;
use crate::arithmetic::{InstructionT, Register};


//pub struct Field<const N_BITS : usize, const N_LIMBS : usize>(Register);

use anyhow::{Result, anyhow};


pub trait FieldRegister<const N_LIMBS : usize> : Sized + Send + Sync {
    const MODULUS : [u16; N_LIMBS];

    /// Returns an element of the field
    /// 
    /// Assumes register is of the correct size
    fn from_raw_register(register : Register) -> Self;

    /// Returns an element of the field
    /// 
    /// Checks that the register is of the correct size
    fn from_register(register : Register) -> Result<Self> {
        if register.len() != N_LIMBS {
            return Err(anyhow!("Invalid register length"));
        }

        Ok(Self::from_raw_register(register))
    }
}
