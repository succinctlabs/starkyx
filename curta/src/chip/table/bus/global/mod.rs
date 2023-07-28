use core::marker::PhantomData;

use super::channel::BusChannel;
use crate::chip::builder::AirBuilder;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::AirParameters;

/// The main bus enforcing the constraint
///
/// Prod(Channels) = 1
#[derive(Clone, Debug)]
pub struct Bus<E> {
    /// The channels of the bus
    channels: Vec<CubicRegister>,
    // The challenge used
    challenge: CubicRegister,
    _marker: PhantomData<E>,
}

impl<E> Bus<E> {
    pub fn new_channel<L: AirParameters<CubicParams = E>>(
        &mut self,
        builder: &mut AirBuilder<L>,
    ) -> BusChannel<L::Field, L::CubicParams> {
        let out_channel = builder.alloc_global::<CubicRegister>();
        self.channels.push(out_channel);
        let accumulator = builder.alloc_extended::<CubicRegister>();
        BusChannel::new(self.challenge, out_channel, accumulator)
    }
}
