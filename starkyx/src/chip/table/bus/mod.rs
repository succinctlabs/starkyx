//! An implementation of an input/output bus based on a permutaton argument
//!
//! The consistency constraints on the bus mean that the every output from the bus has
//! been read from some input.
//!

pub mod channel;
pub mod global;
