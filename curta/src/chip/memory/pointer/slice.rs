use core::marker::PhantomData;

use super::super::value::MemoryValue;
use super::raw::RawPointer;
use super::Pointer;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::AirParameters;

#[derive(Clone, Debug)]
pub struct RawSlice {
    challenge: CubicRegister,
}

#[derive(Clone, Debug)]
pub struct Slice<T> {
    raw: RawSlice,
    challenges: ArrayRegister<CubicRegister>,
    _marker: PhantomData<T>,
}

impl<V: MemoryValue> Slice<V> {
    pub fn new(raw_slice: RawSlice, challenges: ArrayRegister<CubicRegister>) -> Self {
        Self {
            raw: raw_slice,
            challenges,
            _marker: PhantomData,
        }
    }

    pub fn get(&self, idx: usize) -> Pointer<V> {
        let raw = self.raw.get(idx);
        Pointer::new(raw, self.challenges)
    }

    pub fn get_at(&self, idx: ElementRegister) -> Pointer<V> {
        let raw = self.raw.get_at(idx);
        Pointer::new(raw, self.challenges)
    }

    pub fn get_at_shifted(&self, idx: ElementRegister, shift: i32) -> Pointer<V> {
        let raw = self.raw.get_at_shifted(idx, shift);
        Pointer::new(raw, self.challenges)
    }
}

impl RawSlice {
    pub(crate) fn get(&self, idx: usize) -> RawPointer {
        assert!(idx <= i32::MAX as usize);
        RawPointer::new(self.challenge, None, Some(idx as i32))
    }

    pub(crate) fn new<L: AirParameters>(builder: &mut AirBuilder<L>) -> Self {
        let challenge = builder.alloc_challenge();

        Self { challenge }
    }

    pub(crate) fn get_at(&self, idx: ElementRegister) -> RawPointer {
        RawPointer::new(self.challenge, Some(idx), None)
    }

    pub(crate) fn get_at_shifted(&self, idx: ElementRegister, shift: i32) -> RawPointer {
        RawPointer::new(self.challenge, Some(idx), Some(shift))
    }
}
