#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::new_without_default)]
#![feature(bigint_helper_methods)]

extern crate alloc;

pub mod air;
pub mod challenger;
pub mod chip;
pub mod math;
pub mod maybe_rayon;
pub mod polynomial;
pub mod stark;
pub mod trace;

#[cfg(feature = "plonky2")]
pub mod plonky2;
