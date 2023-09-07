use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

use crate::chip::hash::CurtaBytes;
use crate::math::prelude::CubicParameters;

#[derive(Debug, Clone)]
pub struct BLAKE2BBuilderGadget<F, E, const D: usize> {
    pub padded_messages: Vec<Target>,
    pub digests: Vec<Target>,
    pub chunk_sizes: Vec<usize>,
    _marker: PhantomData<(F, E)>,
}

pub trait BLAKE2BBuilder<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> {
    type Gadget;

    fn init_blake2b(&mut self) -> Self::Gadget;

    fn blake2b<const N: usize>(
        &mut self,
        padded_message: &CurtaBytes<N>,
        message_len: usize,
        gadget: &mut Self::Gadget,
    ) -> CurtaBytes<32>;

    fn constrain_blake2b_gadget<C: GenericConfig<D, F = F, FE = F::Extension> + 'static + Clone>(
        &mut self,
        gadget: Self::Gadget,
    ) where
        C::Hasher: AlgebraicHasher<F>;
}

impl<F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> BLAKE2BBuilder<F, E, D>
    for CircuitBuilder<F, D>
{
    type Gadget = BLAKE2BBuilderGadget<F, E, D>;

    fn init_blake2b(&mut self) -> Self::Gadget {
        BLAKE2BBuilderGadget {
            padded_messages: Vec::new(),
            digests: Vec::new(),
            chunk_sizes: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn blake2b<const N: usize>(
        &mut self,
        padded_message: &CurtaBytes<N>,
        message_len: usize,
        gadget: &mut Self::Gadget,
    ) -> CurtaBytes<32> {
        gadget.padded_messages.extend_from_slice(&padded_message.0);
        let digest_bytes = self.add_virtual_target_arr::<32>();
        let hint = BLAKE2BHintGenerator::new(&padded_message.0, message_len, digest_bytes);
        self.add_simple_generator(hint);
        gadget.digests.extend_from_slice(&digest_bytes);
        gadget.chunk_sizes.push(N / 64);
        CurtaBytes(digest_bytes)
    }
}
