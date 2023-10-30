use core::fmt::Debug;

use plonky2::field::extension::Extendable;
use plonky2::field::packable::Packable;
use plonky2::hash::hash_types::RichField;
use serde::de::DeserializeOwned;
use serde::Serialize;

use self::parser::global::{GlobalRecursiveStarkParser, GlobalStarkParser};
use self::parser::{RecursiveStarkParser, StarkParser};
use crate::air::RAir;

pub mod cubic;
pub mod field;
pub mod parser;
pub mod stark;
pub mod trace;

/// an air that can generate constraints for the Starky proving system.
pub trait StarkyAir<F: RichField + Extendable<D>, const D: usize>:
    for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>
    + for<'a> RAir<StarkParser<'a, F, F::Extension, F::Extension, D, D>>
    + for<'a> RAir<GlobalStarkParser<'a, F, F, F, D, 1>>
    + 'static
    + Debug
    + Send
    + Sync
    + Serialize
    + DeserializeOwned
{
}

/// an air that can be verified recursively inside a Plonky2 circuit.
pub trait Plonky2Air<F: RichField + Extendable<D>, const D: usize>:
    StarkyAir<F, D>
    + for<'a> RAir<RecursiveStarkParser<'a, F, D>>
    + for<'a> RAir<GlobalRecursiveStarkParser<'a, F, D>>
{
}

impl<F: RichField + Extendable<D>, const D: usize, T> StarkyAir<F, D> for T where
    T: for<'a> RAir<StarkParser<'a, F, F, <F as Packable>::Packing, D, 1>>
        + for<'a> RAir<StarkParser<'a, F, F::Extension, F::Extension, D, D>>
        + for<'a> RAir<GlobalStarkParser<'a, F, F, F, D, 1>>
        + 'static
        + Debug
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
{
}

impl<F: RichField + Extendable<D>, const D: usize, T> Plonky2Air<F, D> for T where
    T: StarkyAir<F, D>
        + for<'a> RAir<RecursiveStarkParser<'a, F, D>>
        + for<'a> RAir<GlobalRecursiveStarkParser<'a, F, D>>
{
}
