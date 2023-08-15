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
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::RegisterSerializable;
use crate::chip::AirParameters;

/// The main bus enforcing the constraint
///
/// Prod(Channels) = 1
#[derive(Clone, Debug)]
pub struct Bus<E> {
    /// The channels of the bus
    channels: Vec<CubicRegister>,
    // Public inputs to the bus
    global_inputs: Vec<CubicRegister>,
    // The challenge used
    challenge: CubicRegister,
    _marker: PhantomData<E>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_bus(&mut self) -> Bus<L::CubicParams> {
        let challenge = self.alloc_challenge::<CubicRegister>();
        Bus {
            channels: Vec::new(),
            global_inputs: Vec::new(),
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
    ) -> usize {
        let out_channel = builder.alloc_global::<CubicRegister>();
        self.channels.push(out_channel);
        let accumulator = builder.alloc_extended::<CubicRegister>();
        let channel = BusChannel::new(self.challenge, out_channel, accumulator);
        let index = builder.bus_channels.len();
        builder.bus_channels.push(channel);
        index
    }

    pub fn insert_global_value(&mut self, register: &CubicRegister) {
        match register.register() {
            MemorySlice::Global(..) => {
                self.global_inputs.push(*register);
            }
            MemorySlice::Public(..) => {
                self.global_inputs.push(*register);
            }
            _ => panic!("Expected public or global register"),
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Sample;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::register::bit::BitRegister;
    use crate::chip::register::Register;
    use crate::chip::AirParameters;
    use crate::math::extension::cubic::element::CubicElement;
    use crate::math::prelude::*;

    #[derive(Debug, Clone)]
    struct BusTest;

    impl AirParameters for BusTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = 7;
        const EXTENDED_COLUMNS: usize = 12;

        type Instruction = EmptyInstruction<GoldilocksField>;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_bus() {
        type L = BusTest;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;

        let num_public_vals = 1000;

        let mut builder = AirBuilder::<L>::new();
        let x_1 = builder.alloc::<CubicRegister>();
        let x_2 = builder.alloc::<CubicRegister>();

        let bit = builder.alloc::<BitRegister>();

        let mut bus = builder.new_bus();
        let channel_idx = bus.new_channel(&mut builder);

        builder.input_to_bus_filtered(channel_idx, x_1, bit.expr());
        builder.output_from_bus(channel_idx, x_2);

        let x_pub = builder.alloc_array_public::<CubicRegister>(num_public_vals);
        for x in x_pub.iter() {
            bus.insert_global_value(&x);
        }

        builder.constrain_bus(bus);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new(&air);
        let writer = generator.new_writer();

        for (i, x) in x_pub.iter().enumerate() {
            let x_pub_val = CubicElement([GoldilocksField::rand(); 3]);
            writer.write(&x, &x_pub_val, 0);
            writer.write(&x_2, &x_pub_val, L::num_rows() - i - 1);
            writer.write(&bit, &F::ZERO, i);
        }
        for i in num_public_vals..L::num_rows() {
            let a = CubicElement([GoldilocksField::rand(); 3]);
            writer.write(&bit, &F::ONE, i);
            writer.write(&x_1, &a, i);
            writer.write(&x_2, &a, L::num_rows() - i - 1);
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(L::num_rows());

        let public_inputs = writer.0.public.read().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
