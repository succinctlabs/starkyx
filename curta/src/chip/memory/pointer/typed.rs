use core::marker::PhantomData;

use super::raw::RawPointer;
use crate::chip::register::cubic::CubicRegister;

/// A pointer emulating a mutable reference to a data of register type `T`.
#[derive(Debug, Clone, Copy)]
pub struct Pointer<T> {
    pub raw: RawPointer,
    _marker: PhantomData<T>,
}

impl<T> Pointer<T> {
    pub fn new(raw_ptr: RawPointer) -> Self {
        Self {
            raw: raw_ptr,
            _marker: PhantomData,
        }
    }

    pub(crate) fn from_challenge(challenge: CubicRegister) -> Self {
        Self::new(RawPointer::from_challenge(challenge))
    }
}
