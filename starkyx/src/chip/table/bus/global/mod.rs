pub mod constraint;
pub mod trace;

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use super::channel::BusChannel;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::table::log_derivative::entry::LogEntry;
use crate::chip::AirParameters;

/// A general communication bus enforcing the constraint that very element inserted as input to the
/// bus was also taken as output from the bus.
///
/// The constraints reflecting the bus logic use a random challenge `beta` to assert that the sum
/// `sum_i 1/(beta - input_i) - sum_j 1/(beta -output_j) = 0` where `input_i` are the inputs to the
/// bus and `output_j` are the outputs from the bus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bus<T, E> {
    /// The channels of the bus
    channels: Vec<CubicRegister>,
    // Public inputs to the bus
    global_entries: Vec<LogEntry<T>>,
    // Accumulators for the partial sums of global entry values.
    global_accumulators: ArrayRegister<CubicRegister>,
    /// The total accumulated value of the global entries.
    global_value: CubicRegister,
    // The challenge used
    challenge: CubicRegister,
    _marker: PhantomData<E>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_bus(&mut self) -> Bus<CubicRegister, L::CubicParams> {
        let challenge = self.alloc_challenge::<CubicRegister>();
        let global_value = self.alloc_global::<CubicRegister>();
        Bus {
            channels: Vec::new(),
            global_entries: Vec::new(),
            global_accumulators: ArrayRegister::uninitialized(),
            global_value,
            challenge,
            _marker: PhantomData,
        }
    }

    pub fn constrain_bus(&mut self, bus: Bus<CubicRegister, L::CubicParams>) {
        self.buses.push(bus.clone());
    }

    pub fn register_bus_constraint(&mut self, index: usize) {
        let global_entries_len = self.buses[index].global_entries.len();
        let global_accumulators = self.alloc_array_global::<CubicRegister>(global_entries_len / 2);
        self.buses[index].global_accumulators = global_accumulators;
        self.global_constraints
            .push(self.buses[index].clone().into());
    }
}

impl<T: EvalCubic, E: Clone> Bus<T, E> {
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

    pub fn insert_global_value(&mut self, value: &T) {
        match value.register() {
            MemorySlice::Global(..) => {
                self.global_entries.push(LogEntry::Input(*value));
            }
            MemorySlice::Public(..) => {
                self.global_entries.push(LogEntry::Input(*value));
            }
            _ => panic!("Expected public or global register"),
        }
    }

    pub fn insert_global_value_with_multiplicity(
        &mut self,
        value: &T,
        multiplicity: ElementRegister,
    ) {
        match value.register() {
            MemorySlice::Global(..) => {
                self.global_entries
                    .push(LogEntry::InputMultiplicity(*value, multiplicity));
            }
            MemorySlice::Public(..) => {
                self.global_entries
                    .push(LogEntry::InputMultiplicity(*value, multiplicity));
            }
            _ => panic!("Expected public or global register"),
        }
    }

    pub fn output_global_value(&mut self, value: &T) {
        match value.register() {
            MemorySlice::Global(..) => {
                self.global_entries.push(LogEntry::Output(*value));
            }
            MemorySlice::Public(..) => {
                self.global_entries.push(LogEntry::Output(*value));
            }
            _ => panic!("Expected public or global register"),
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Sample;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::register::bit::BitRegister;
    use crate::math::extension::cubic::element::CubicElement;
    use crate::math::prelude::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct BusTest;

    impl AirParameters for BusTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = 7;
        const EXTENDED_COLUMNS: usize = 12;

        type Instruction = EmptyInstruction<GoldilocksField>;
    }

    #[test]
    fn test_bus_global() {
        type L = BusTest;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;

        let num_public_in = 100;
        let num_public_out = 50;
        assert!(num_public_out < num_public_in);

        let mut builder = AirBuilder::<L>::new();
        let x_1 = builder.alloc::<CubicRegister>();
        let x_2 = builder.alloc::<CubicRegister>();

        let bit = builder.alloc::<BitRegister>();

        let mut bus = builder.new_bus();
        let channel_idx = bus.new_channel(&mut builder);

        builder.input_to_bus_filtered(channel_idx, x_1, bit);
        builder.output_from_bus(channel_idx, x_2);

        let x_in_pub = builder.alloc_array_public::<CubicRegister>(num_public_in);
        let x_out_pub = builder.alloc_array_public::<CubicRegister>(num_public_out);
        for x in x_in_pub.iter() {
            bus.insert_global_value(&x);
        }
        for x in x_out_pub.iter() {
            bus.output_global_value(&x);
        }

        builder.constrain_bus(bus);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        for (i, x) in x_in_pub.iter().enumerate() {
            let x_pub_val = CubicElement([GoldilocksField::rand(); 3]);
            writer.write(&x, &x_pub_val, 0);
            if i < num_public_out {
                writer.write(&x_out_pub.get(i), &x_pub_val, 0);
            } else {
                let row = i - num_public_out;
                writer.write(&x_2, &x_pub_val, num_rows - row - 1);
                writer.write(&bit, &F::ZERO, row);
            }
        }
        for i in (num_public_in - num_public_out)..num_rows {
            let a = CubicElement([GoldilocksField::rand(); 3]);
            writer.write(&bit, &F::ONE, i);
            writer.write(&x_1, &a, i);
            writer.write(&x_2, &a, num_rows - i - 1);
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(num_rows);

        let public_inputs = writer.0.public.read().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
