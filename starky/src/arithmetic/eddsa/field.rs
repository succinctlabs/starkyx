use num::{BigUint, One, Zero};

use crate::arithmetic::register::{CellType, DataRegister, U16Array};
use crate::arithmetic::Register;

pub trait FieldParameters<const N_LIMBS: usize>: Send + Sync + Copy + 'static {
    const MODULUS: [u16; N_LIMBS];

    fn modulus_biguint() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::MODULUS.iter().enumerate() {
            modulus += BigUint::from(*limb) << (16 * i);
        }
        modulus
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FieldRegister<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> {
    array: U16Array<N_LIMBS>,
    _marker: core::marker::PhantomData<P>,
}

impl<const N_LIMBS: usize, P: FieldParameters<N_LIMBS>> DataRegister for FieldRegister<P, N_LIMBS> {
    const CELL: Option<CellType> = Some(CellType::U16);

    fn from_raw_register(register: Register) -> Self {
        Self {
            array: U16Array::from_raw_register(register),
            _marker: core::marker::PhantomData,
        }
    }

    fn register(&self) -> &Register {
        self.array.register()
    }

    fn register_mut(&mut self) -> &mut Register {
        self.array.register_mut()
    }

    fn size_of() -> usize {
        N_LIMBS
    }

    fn into_raw_register(self) -> Register {
        self.array.into_raw_register()
    }
}

/// The parameters for the Fp25519 field of modulues 2^255-19.
#[derive(Debug, Clone, Copy)]
pub struct Fp25519Param;

pub type Fp25519 = FieldRegister<Fp25519Param, 16>;

impl FieldParameters<16> for Fp25519Param {
    const MODULUS: [u16; 16] = [
        65517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
        65535, 65535, 32767,
    ];

    fn modulus_biguint() -> BigUint {
        (BigUint::one() << 255) - BigUint::from(19u32)
    }
}
