#![allow(incomplete_features)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![cfg_attr(not(feature = "std"), no_std)]
#![feature(test)]

extern crate alloc;

mod get_challenges;


pub mod arithmetic;
pub mod benchmark;
pub mod config;
pub mod constraint_consumer;
pub mod permutation;
pub mod proof;
pub mod prover;
pub mod recursive_verifier;
pub mod stark;
pub mod stark_testing;
pub mod util;
pub mod vanishing_poly;
pub mod vars;
pub mod verifier;

#[cfg(test)]
pub mod fibonacci_stark;
