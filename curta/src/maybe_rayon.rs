//! A convenient wrapper for parallel iterator trait that links to the Rayon crate.
//! uses the plonky2_maybe_rayon crate to conditionally compile with or without rayon.

pub use plonky2_maybe_rayon::*;
