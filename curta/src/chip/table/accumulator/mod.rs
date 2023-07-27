//! A row accumulator
//!
//!
//!
//!

pub mod constraint;
pub mod trace;

use core::marker::PhantomData;

use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::extension::ExtensionRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::RegisterSerializable;
use crate::chip::AirParameters;

#[derive(Debug, Clone)]
pub struct Accumulator<E> {
    pub(crate) challenges: ArrayRegister<ExtensionRegister<3>>,
    values: Vec<MemorySlice>,
    digest: ExtensionRegister<3>,
    _marker: PhantomData<E>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn accumulate<T: RegisterSerializable>(
        &mut self,
        challenges: &ArrayRegister<ExtensionRegister<3>>,
        values: &[T],
    ) -> ExtensionRegister<3> {
        let values = values
            .into_iter()
            .map(|data| *data.register())
            .collect::<Vec<_>>();
        let total_length = values.iter().map(|data| data.len()).sum::<usize>();
        assert_eq!(
            total_length,
            challenges.len(),
            "Accumulator challenges and values must be the same size"
        );

        let digest = self.alloc_extended::<ExtensionRegister<3>>();

        let accumulator = Accumulator {
            challenges: *challenges,
            values,
            digest,
            _marker: PhantomData,
        };

        self.accumulators.push(accumulator.clone());
        self.constraints.push(accumulator.into());

        digest
    }
}
