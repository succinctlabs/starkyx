//! An implementation of an input/output bus based on a permutaton argument
//!
//! The consistency constraints on the bus mean that the every output from the bus has
//! been read from some input.
//!

use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::plonky2::field::cubic::element::CubicElement;

/// The main bus enforcing the constraint
///
/// Prod(Channels) = 1
#[derive(Clone, Debug)]
pub struct Bus {
    /// The channels of the bus
    channels: Vec<CubicRegister>,
}

#[derive(Clone, Debug)]
pub struct BusChannel {
    out_channel: CubicRegister,
    accumulator: CubicRegister,
    entries: Vec<CubicRegister>,
}

impl Bus {
    pub fn new_channel<L : AirParameters>(&mut self, builder : &mut AirBuilder<L>) -> BusChannel {
        let out_channel = builder.alloc_global::<CubicRegister>();
        self.channels.push(out_channel);
        let accumulator = builder.alloc_extended::<CubicRegister>();
        BusChannel { out_channel, accumulator, entries: Vec::new() }
    }
}
