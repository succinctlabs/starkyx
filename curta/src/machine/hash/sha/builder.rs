use super::algorithm::SHAir;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::machine::builder::Builder;

pub trait SHABuilder: Builder {
    fn sha<S: SHAir<Self, CYCLE_LENGTH>, const CYCLE_LENGTH: usize>(
        &mut self,
        padded_chunks: &[ArrayRegister<S::Variable>],
        end_bits: &ArrayRegister<BitRegister>,
    ) -> Vec<S::StateVariable> {
        S::sha(self, padded_chunks, end_bits)
    }
}

impl<B: Builder> SHABuilder for B {}
