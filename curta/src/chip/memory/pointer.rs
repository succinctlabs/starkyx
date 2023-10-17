use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::RegisterSerializable;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RawPointer {
    challenge: CubicRegister,
}

impl RawPointer {
    pub(crate) fn from_challenge(challenge: CubicRegister) -> Self {
        Self { challenge }
    }

    pub fn challenge(&self) -> ArrayRegister<CubicRegister> {
        ArrayRegister::from_register_unsafe(*self.challenge.register())
    }
}

#[derive(Debug, Clone, Copy, Hash)]
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
