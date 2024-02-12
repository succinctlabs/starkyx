use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::config::{CurtaConfig, StarkyConfig};
use super::proof::StarkProofTarget;
use super::verifier::{add_virtual_stark_proof, StarkyVerifier};
use super::Starky;
use crate::plonky2::Plonky2Air;

pub trait StarkGadget<
    F: RichField + Extendable<D>,
    C: CurtaConfig<D, F = F, FE = F::Extension>,
    const D: usize,
>
{
    fn add_virtual_stark_proof<A>(
        &mut self,
        stark: &Starky<A>,
        config: &StarkyConfig<C, D>,
    ) -> StarkProofTarget<D>
    where
        A: Plonky2Air<F, D>;

    fn verify_stark_proof<A>(
        &mut self,
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        proof: &StarkProofTarget<D>,
        public_inputs: &[Target],
    ) where
        A: Plonky2Air<F, D>;
}

impl<F: RichField + Extendable<D>, C: CurtaConfig<D, F = F, FE = F::Extension>, const D: usize>
    StarkGadget<F, C, D> for CircuitBuilder<F, D>
{
    fn add_virtual_stark_proof<A>(
        &mut self,
        stark: &Starky<A>,
        config: &StarkyConfig<C, D>,
    ) -> StarkProofTarget<D>
    where
        A: Plonky2Air<F, D>,
    {
        add_virtual_stark_proof(self, stark, config)
    }

    fn verify_stark_proof<A>(
        &mut self,
        config: &StarkyConfig<C, D>,
        stark: &Starky<A>,
        proof: &StarkProofTarget<D>,
        public_inputs: &[Target],
    ) where
        A: Plonky2Air<F, D>,
    {
        StarkyVerifier::verify_circuit(self, config, stark, proof, public_inputs)
    }
}
