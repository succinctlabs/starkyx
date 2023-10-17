use core::marker::PhantomData;

use super::pointer::{Pointer, RawPointer};
use super::value::MemoryValue;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Clone, Debug)]
pub struct RawSlice {
    challenges: ArrayRegister<CubicRegister>,
    idx_challenges: Vec<CubicRegister>,
}

#[derive(Clone, Debug)]
pub struct Slice<T> {
    raw: RawSlice,
    _marker: PhantomData<T>,
}

impl<V: MemoryValue> Slice<V> {
    pub fn get(&self, idx: usize) -> Pointer<V> {
        let raw = self.raw.get(idx);
        Pointer::new(raw)
    }

    pub(crate) fn get_at<L: AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        idx: ElementRegister,
    ) -> Pointer<V> {
        let raw = self.raw.get_at(builder, idx);
        Pointer::new(raw)
    }
}

impl RawSlice {
    fn get(&self, idx: usize) -> RawPointer {
        let challenge = self.idx_challenges.get(idx).expect("Index out of bounds");
        RawPointer::from_challenge(*challenge)
    }

    fn get_at<L: AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        idx: ElementRegister,
    ) -> RawPointer {
        match idx.register() {
            MemorySlice::Local(_, _) => {
                let challenge = builder
                    .accumulate_expressions(&self.challenges, &[L::Field::ONE.into(), idx.expr()]);
                RawPointer::from_challenge(challenge)
            }
            MemorySlice::Public(_, _) => {
                let challenge = builder.accumulate_public_expressions(
                    &self.challenges,
                    &[L::Field::ONE.into(), idx.expr()],
                );
                RawPointer::from_challenge(challenge)
            }
            _ => panic!("Expected local or public register"),
        }
    }
}
