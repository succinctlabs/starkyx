#![allow(incomplete_features)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![cfg_attr(not(feature = "std"), no_std)]
#![feature(test)]
#![feature(const_trait_impl)]
#![feature(const_ops)]


pub mod math;
pub mod maybe_rayon;