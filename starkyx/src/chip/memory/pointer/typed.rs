use core::marker::PhantomData;

use super::raw::RawPointer;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;

/// A pointer emulating a mutable reference to a data of register type `T`.
#[derive(Debug, Clone, Copy)]
pub struct Pointer<T> {
    pub raw: RawPointer,
    pub challenges: ArrayRegister<CubicRegister>,
    _marker: PhantomData<T>,
}

impl<T> Pointer<T> {
    pub fn new(raw_ptr: RawPointer, challenges: ArrayRegister<CubicRegister>) -> Self {
        Self {
            raw: raw_ptr,
            challenges,
            _marker: PhantomData,
        }
    }

    pub(crate) fn from_challenges(
        raw_ptr_challenge_powers: ArrayRegister<CubicRegister>,
        compression_challenges: ArrayRegister<CubicRegister>,
    ) -> Self {
        Self::new(
            RawPointer::from_challenge(raw_ptr_challenge_powers),
            compression_challenges,
        )
    }
}
