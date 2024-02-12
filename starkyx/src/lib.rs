#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::new_without_default)]
#![feature(bigint_helper_methods)]

extern crate alloc;

pub mod air;
pub mod chip;
pub mod machine;
pub mod math;
pub mod maybe_rayon;
pub mod polynomial;
pub mod trace;
pub mod utils;

#[cfg(feature = "plonky2")]
pub mod plonky2;

pub mod prelude {
    pub use crate::air::parser::AirParser;
    pub use crate::air::AirConstraint;
    pub use crate::chip::instruction::empty::EmptyInstruction;
    pub use crate::chip::trace::writer::data::AirWriterData;
    pub use crate::chip::trace::writer::AirWriter;
    pub use crate::chip::AirParameters;
    pub use crate::machine::builder::Builder;
    pub use crate::machine::bytes::builder::BytesBuilder;
    pub use crate::machine::bytes::stark::ByteStark;
    pub use crate::machine::emulated::builder::EmulatedBuilder;
    pub use crate::machine::emulated::stark::EmulatedStark;
    pub use crate::machine::stark::builder::StarkBuilder;
    pub use crate::machine::stark::Stark;
    pub use crate::math::prelude::*;
    pub use crate::maybe_rayon::*;
}
