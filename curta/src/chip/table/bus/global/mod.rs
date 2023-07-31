//! Bus constraint
//!
//! The globa bus constraint enforeces consistency between the channels of the bus.
//! Namely, the constraint:
//!
//! \prod_{i=1}^n channel_i = 1
//!

pub mod constraint;

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

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_bus(&mut self) -> Bus<L::CubicParams> {
        let challenge = self.alloc_challenge::<CubicRegister>();
        Bus {
            channels: Vec::new(),
            challenge,
            _marker: PhantomData,
        }
    }

    pub fn constrain_bus(&mut self, bus: Bus<L::CubicParams>) {
        self.constraints.push(bus.into());
    }
}

impl<E: Clone> Bus<E> {
    pub fn new_channel<L: AirParameters<CubicParams = E>>(
        &mut self,
        builder: &mut AirBuilder<L>,
    ) -> BusChannel<L::Field, L::CubicParams> {
        let out_channel = builder.alloc_global::<CubicRegister>();
        self.channels.push(out_channel);
        let accumulator = builder.alloc_extended::<CubicRegister>();
        let channel = BusChannel::new(self.challenge, out_channel, accumulator);
        builder.bus_channels.push(channel.clone());
        builder.constraints.push(channel.clone().into());
        channel
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Sample;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::AirParameters;
    use crate::math::extension::cubic::element::CubicElement;

    #[derive(Debug, Clone)]
    struct BusTest;

    impl AirParameters for BusTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = 6;
        const EXTENDED_COLUMNS: usize = 12;

        type Instruction = EmptyInstruction<GoldilocksField>;

        fn num_rows_bits() -> usize {
            10
        }
    }

    #[test]
    fn test_bus() {
        type L = BusTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_1 = builder.alloc::<CubicRegister>();
        let x_2 = builder.alloc::<CubicRegister>();

        let mut bus = builder.new_bus();
        let mut channel = bus.new_channel(&mut builder);

        builder.input_to_bus(&mut channel, x_1);
        builder.output_from_bus(&mut channel, x_2);

        builder.constrain_bus(bus);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();
        for i in 0..L::num_rows() {
            let a = CubicElement([GoldilocksField::rand(); 3]);
            writer.write(&x_1, &a, L::num_rows() - 1 - i);
            writer.write(&x_2, &a, i);
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
