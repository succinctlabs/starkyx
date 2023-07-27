use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

use super::config::StarkyConfig;
use super::proof::StarkProofTarget;
use super::verifier::{add_virtual_stark_proof, StarkyVerifier};
use super::Starky;
use crate::air::RAir;
use crate::plonky2::parser::RecursiveStarkParser;

pub trait StarkGadget<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F, FE = F::Extension>,
    const D: usize,
>
{
    fn add_virtual_stark_proof<A, const COLUMNS: usize>(
        &mut self,
        stark: &Starky<A, COLUMNS>,
        config: &StarkyConfig<F, C, D>,
    ) -> StarkProofTarget<D>
    where
        C::Hasher: AlgebraicHasher<F>,
        A: for<'a> RAir<RecursiveStarkParser<'a, F, D>>;

    fn verify_stark_proof<A, const COLUMNS: usize>(
        &mut self,
        config: &StarkyConfig<F, C, D>,
        stark: &Starky<A, COLUMNS>,
        proof: StarkProofTarget<D>,
        public_inputs: &[Target],
    ) where
        C::Hasher: AlgebraicHasher<F>,
        A: for<'a> RAir<RecursiveStarkParser<'a, F, D>>;
}

impl<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F, FE = F::Extension>,
        const D: usize,
    > StarkGadget<F, C, D> for CircuitBuilder<F, D>
{
    fn add_virtual_stark_proof<A, const COLUMNS: usize>(
        &mut self,
        stark: &Starky<A, COLUMNS>,
        config: &StarkyConfig<F, C, D>,
    ) -> StarkProofTarget<D>
    where
        C::Hasher: AlgebraicHasher<F>,
        A: for<'a> RAir<RecursiveStarkParser<'a, F, D>>,
    {
        add_virtual_stark_proof(self, stark, config)
    }

    fn verify_stark_proof<A, const COLUMNS: usize>(
        &mut self,
        config: &StarkyConfig<F, C, D>,
        stark: &Starky<A, COLUMNS>,
        proof: StarkProofTarget<D>,
        public_inputs: &[Target],
    ) where
        C::Hasher: AlgebraicHasher<F>,
        A: for<'a> RAir<RecursiveStarkParser<'a, F, D>>,
    {
        StarkyVerifier::verify_circuit(self, config, stark, proof, public_inputs)
    }
}
