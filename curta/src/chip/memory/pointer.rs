use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::chip::register::cubic::CubicRegister;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RawPointer {
    challenge: CubicRegister,
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct Pointer<T> {
    ptr: RawPointer,
    _marker: PhantomData<T>,
}

impl<T> Pointer<T> {
    pub fn new(ptr: RawPointer) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn from_challenge(challenge: CubicRegister) -> Self {
        Self::new(RawPointer { challenge })
    }

    pub fn raw_ptr(&self) -> RawPointer {
        self.ptr
    }
}
