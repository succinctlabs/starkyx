pub mod constraint;
pub mod trace;

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::chip::builder::AirBuilder;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::{CubicRegister, EvalCubic};
use crate::chip::register::element::ElementRegister;
use crate::chip::table::log_derivative::entry::LogEntry;
use crate::chip::AirParameters;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusChannel<T, E> {
    pub out_channel: CubicRegister,
    table_accumulator: CubicRegister,
    challenge: CubicRegister,
    entries: Vec<LogEntry<T>>,
    row_accumulators: Vec<CubicRegister>,
    _marker: PhantomData<E>,
}

impl<L: AirParameters> AirBuilder<L> {
    #[inline]
    fn add_accumulators(&mut self, channel_idx: usize) {
        let length = self.bus_channels[channel_idx].entries.len();
        if length % 2 == 0 {
            let acc_sum = self.alloc_extended::<CubicRegister>();
            self.bus_channels[channel_idx]
                .row_accumulators
                .push(acc_sum);
        }
    }

    pub fn input_to_bus(&mut self, channel_idx: usize, value: CubicRegister) {
        self.bus_channels[channel_idx].input(value);
        self.add_accumulators(channel_idx);
    }

    pub fn input_to_bus_filtered(
        &mut self,
        channel_idx: usize,
        value: CubicRegister,
        filter: BitRegister,
    ) {
        self.bus_channels[channel_idx].input_filtered(value, filter);
        self.add_accumulators(channel_idx);
    }

    pub fn input_to_bus_with_multiplicity(
        &mut self,
        channel_idx: usize,
        value: CubicRegister,
        multiplicity: ElementRegister,
    ) {
        self.bus_channels[channel_idx].input_with_multiplicity(value, multiplicity);
        self.add_accumulators(channel_idx);
    }

    pub fn output_from_bus(&mut self, channel_idx: usize, value: CubicRegister) {
        self.bus_channels[channel_idx].output(value);
        self.add_accumulators(channel_idx);
    }

    pub fn output_from_bus_filtered(
        &mut self,
        channel_idx: usize,
        value: CubicRegister,
        filter: BitRegister,
    ) {
        self.bus_channels[channel_idx].output_filtered(value, filter);
        self.add_accumulators(channel_idx);
    }

    pub fn output_from_bus_with_multiplicity(
        &mut self,
        channel_idx: usize,
        value: CubicRegister,
        multiplicity: ElementRegister,
    ) {
        self.bus_channels[channel_idx].output_with_multiplicity(value, multiplicity);
        self.add_accumulators(channel_idx);
    }
}

impl<T: EvalCubic, E> BusChannel<T, E> {
    pub fn new(
        challenge: CubicRegister,
        out_channel: CubicRegister,
        table_accumulator: CubicRegister,
    ) -> Self {
        Self {
            challenge,
            out_channel,
            table_accumulator,
            entries: Vec::new(),
            row_accumulators: Vec::new(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn input(&mut self, value: T) {
        let entry = LogEntry::Input(value);
        self.entries.push(entry)
    }

    #[inline]
    pub fn input_filtered(&mut self, value: T, filter: BitRegister) {
        self.input_with_multiplicity(value, filter.as_element())
    }

    #[inline]
    pub fn input_with_multiplicity(&mut self, value: T, multiplicity: ElementRegister) {
        self.entries
            .push(LogEntry::InputMultiplicity(value, multiplicity));
    }

    #[inline]
    pub fn output(&mut self, value: T) {
        self.entries.push(LogEntry::Output(value))
    }

    #[inline]
    pub fn output_filtered(&mut self, value: T, filter: BitRegister) {
        self.output_with_multiplicity(value, filter.as_element())
    }

    #[inline]
    pub fn output_with_multiplicity(&mut self, value: T, multiplicity: ElementRegister) {
        self.entries
            .push(LogEntry::OutputMultiplicity(value, multiplicity))
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Sample;

    use super::*;
    use crate::chip::arithmetic::expression::ArithmeticExpression;
    use crate::chip::builder::tests::*;
    use crate::chip::register::Register;
    use crate::math::extension::cubic::element::CubicElement;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct BusChannelTest;

    impl AirParameters for BusChannelTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = 12;
        const EXTENDED_COLUMNS: usize = 21;

        type Instruction = EmptyInstruction<GoldilocksField>;
    }

    #[test]
    fn test_bus_channel() {
        type L = BusChannelTest;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_1 = builder.alloc::<CubicRegister>();
        let x_2 = builder.alloc::<CubicRegister>();
        let x_3 = builder.alloc::<CubicRegister>();
        let x_4 = builder.alloc::<CubicRegister>();

        let beta = builder.alloc_challenge::<CubicRegister>();
        let out_channel = builder.alloc_global::<CubicRegister>();
        let accumulator = builder.alloc_extended::<CubicRegister>();
        let channel = BusChannel::new(beta, out_channel, accumulator);
        builder.bus_channels.push(channel);

        builder.input_to_bus(0, x_1);
        builder.input_to_bus(0, x_3);
        builder.output_from_bus(0, x_2);
        builder.output_from_bus(0, x_4);

        let zero = ArithmeticExpression::<F>::zero();

        let [c_0, c_1, c_2] = out_channel.as_base_array();
        builder.assert_expressions_equal(c_0.expr(), zero.clone());
        builder.assert_expressions_equal(c_1.expr(), zero.clone());
        builder.assert_expressions_equal(c_2.expr(), zero);

        let (air, trace_data) = builder.build();

        let num_rows = 1 << 10;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();
        for i in 0..num_rows {
            let a = CubicElement([GoldilocksField::rand(); 3]);
            let b = CubicElement([GoldilocksField::rand(); 3]);
            writer.write(&x_1, &a, num_rows - 1 - i);
            writer.write(&x_2, &a, i);
            writer.write(&x_3, &b, i);
            writer.write(&x_4, &b, i);
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(num_rows);

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
