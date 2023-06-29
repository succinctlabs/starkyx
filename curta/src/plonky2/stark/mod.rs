//! A vartation of Starky that includes random verifier challenges (RAIR)
//!
//!

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::config::GenericConfig;

use self::config::StarkyConfig;
use super::parser::{RecursiveStarkParser, StarkParser};
use crate::air::Air;
use crate::stark::Stark;

pub mod config;
pub mod proof;
pub mod prover;
pub mod verifier;

pub trait Plonky2Stark<F: RichField + Extendable<D>, const D: usize> {
    type Air;

    fn air(&self) -> &Self::Air;

    /// Columns for each round
    fn round_lengths(&self) -> &[usize];

    /// The number of challenges per round
    fn num_challenges(&self, round: usize) -> usize;
}

impl<'a, T, F, C: GenericConfig<D, F = F>, FE, P, const D: usize, const D2: usize>
    Stark<StarkParser<'a, F, FE, P, D, D2>, StarkyConfig<F, C, D>> for T
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
    T: Plonky2Stark<F, D>,
    T::Air: Air<StarkParser<'a, F, FE, P, D, D2>>,
{
    type Air = T::Air;

    fn air(&self) -> &Self::Air {
        self.air()
    }

    fn round_lengths(&self) -> &[usize] {
        self.round_lengths()
    }

    fn num_challenges(&self, round: usize) -> usize {
        self.num_challenges(round)
    }
}

impl<'a, T, F, C: GenericConfig<D, F = F>, const D: usize>
    Stark<RecursiveStarkParser<'a, F, D>, StarkyConfig<F, C, D>> for T
where
    F: RichField + Extendable<D>,
    T: Plonky2Stark<F, D>,
    T::Air: Air<RecursiveStarkParser<'a, F, D>>,
{
    type Air = T::Air;

    fn air(&self) -> &Self::Air {
        self.air()
    }

    fn round_lengths(&self) -> &[usize] {
        self.round_lengths()
    }

    fn num_challenges(&self, round: usize) -> usize {
        self.num_challenges(round)
    }
}
