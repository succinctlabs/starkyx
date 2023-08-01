//! U32 Channel

use super::opcode::U32Opcode;
use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::table::bus::global::Bus;
use crate::chip::AirParameters;

pub struct U32Channel {
    pub challenges: ArrayRegister<CubicRegister>,
    pub channel_idx: usize,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_u32_channel(
        &mut self,
        challenges: &ArrayRegister<CubicRegister>,
        bus: &mut Bus<L::CubicParams>,
    ) -> U32Channel {
        assert_eq!(challenges.len(), 4);
        let channel_idx = bus.new_channel(self);

        U32Channel {
            challenges: *challenges,
            channel_idx,
        }
    }

    pub fn input_to_u32_channel(&mut self, opcode: &U32Opcode, channel: &U32Channel) {
        let digest = self.accumulate(
            &channel.challenges,
            &[opcode.id, opcode.a, opcode.b, opcode.result],
        );
        self.input_to_bus(channel.channel_idx, digest);
    }

    pub fn input_to_u32_channel_filtered(
        &mut self,
        opcode: &U32Opcode,
        filter: ArithmeticExpression<L::Field>,
        channel: &&U32Channel,
    ) {
        let digest = self.accumulate(
            &channel.challenges,
            &[opcode.id, opcode.a, opcode.b, opcode.result],
        );
        self.input_to_bus_filtered(channel.channel_idx, digest, filter);
    }

    pub fn output_from_u32_channel(&mut self, opcode: &U32Opcode, channel: &U32Channel) {
        let digest = self.accumulate(
            &channel.challenges,
            &[opcode.id, opcode.a, opcode.b, opcode.result],
        );
        self.output_from_bus(channel.channel_idx, digest);
    }

    pub fn output_from_u32_channel_filtered(
        &mut self,
        opcode: &U32Opcode,
        filter: ArithmeticExpression<L::Field>,
        channel: &&U32Channel,
    ) {
        let digest = self.accumulate(
            &channel.challenges,
            &[opcode.id, opcode.a, opcode.b, opcode.result],
        );
        self.output_from_bus_filtered(channel.channel_idx, digest, filter);
    }
}
