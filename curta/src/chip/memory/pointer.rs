use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::chip::register::cubic::CubicRegister;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RawPointer {
    challenge: CubicRegister,
}

impl RawPointer {
    pub(crate) fn from_challenge(challenge: CubicRegister) -> Self {
        Self { challenge }
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
