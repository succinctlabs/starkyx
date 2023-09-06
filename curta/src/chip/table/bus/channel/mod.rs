pub mod constraint;
pub mod entry;
pub mod trace;

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use self::entry::Entry;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusChannel<F, E> {
    pub out_channel: CubicRegister,
    table_accumulator: CubicRegister,
    challenge: CubicRegister,
    entries: Vec<Entry<F>>,
    entry_values: Vec<CubicRegister>,
    row_acc_product: Vec<CubicRegister>,
    _marker: PhantomData<E>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn input_to_bus(&mut self, channel_idx: usize, register: CubicRegister) {
        let entry_values = self.alloc_extended();
        self.bus_channels[channel_idx].input(register);
        self.bus_channels[channel_idx]
            .entry_values
            .push(entry_values);
        let length = self.bus_channels[channel_idx].entries.len();
        if length % 2 == 0 {
            let product = self.alloc_extended::<CubicRegister>();
            self.bus_channels[channel_idx].row_acc_product.push(product);
        }
    }

    pub fn input_to_bus_filtered(
        &mut self,
        channel_idx: usize,
        register: CubicRegister,
        filter: ArithmeticExpression<L::Field>,
    ) {
        let entry_values = self.alloc_extended();
        let bus = &mut self.bus_channels[channel_idx];
        bus.input_filtered(register, filter);
        bus.entry_values.push(entry_values);
        let len = bus.entries.len();
        if len % 2 == 0 {
            let product = self.alloc_extended::<CubicRegister>();
            self.bus_channels[channel_idx].row_acc_product.push(product);
        }
    }

    pub fn output_from_bus(&mut self, channel_idx: usize, register: CubicRegister) {
        let entry_values = self.alloc_extended();
        self.bus_channels[channel_idx].output(register);
        self.bus_channels[channel_idx]
            .entry_values
            .push(entry_values);
        let len = self.bus_channels[channel_idx].entries.len();
        if len % 2 == 0 {
            let product = self.alloc_extended::<CubicRegister>();
            self.bus_channels[channel_idx].row_acc_product.push(product);
        }
    }

    pub fn output_from_bus_filtered(
        &mut self,
        channel_idx: usize,
        register: CubicRegister,
        filter: ArithmeticExpression<L::Field>,
    ) {
        let entry_values = self.alloc_extended();
        self.bus_channels[channel_idx].output_filtered(register, filter);
        self.bus_channels[channel_idx]
            .entry_values
            .push(entry_values);
        let len = self.bus_channels[channel_idx].entries.len();
        if len % 2 == 0 {
            let product = self.alloc_extended::<CubicRegister>();
            self.bus_channels[channel_idx].row_acc_product.push(product);
        }
    }
}

impl<F: Field, E> BusChannel<F, E> {
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
            entry_values: Vec::new(),
            row_acc_product: Vec::new(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn input(&mut self, register: CubicRegister) {
        self.input_filtered(register, ArithmeticExpression::one());
    }

    #[inline]
    pub fn input_filtered(&mut self, register: CubicRegister, filter: ArithmeticExpression<F>) {
        self.entries.push(Entry::Input(register, filter));
    }

    #[inline]
    pub fn output(&mut self, register: CubicRegister) {
        self.output_filtered(register, ArithmeticExpression::one())
    }

    #[inline]
    pub fn output_filtered(&mut self, register: CubicRegister, filter: ArithmeticExpression<F>) {
        self.entries.push(Entry::Output(register, filter));
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Sample;

    use super::*;
    use crate::chip::arithmetic::expression::ArithmeticExpression;
    use crate::chip::builder::tests::*;
    use crate::chip::register::Register;
    use crate::chip::AirParameters;
    use crate::math::extension::cubic::element::CubicElement;

    #[derive(Debug, Clone)]
    struct BusChannelTest;

    impl AirParameters for BusChannelTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_FREE_COLUMNS: usize = 12;
        const EXTENDED_COLUMNS: usize = 21;

        type Instruction = EmptyInstruction<GoldilocksField>;

        fn num_rows_bits() -> usize {
            10
        }
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
        builder.bus_channels.push(channel.clone());

        builder.input_to_bus(0, x_1);
        builder.input_to_bus(0, x_3);
        builder.output_from_bus(0, x_2);
        builder.output_from_bus(0, x_4);

        let one = ArithmeticExpression::<F>::one();
        let zero = ArithmeticExpression::<F>::zero();

        let [c_0, c_1, c_2] = out_channel.as_base_array();
        builder.assert_expressions_equal(c_0.expr(), one);
        builder.assert_expressions_equal(c_1.expr(), zero.clone());
        builder.assert_expressions_equal(c_2.expr(), zero);

        let (air, trace_data) = builder.build();

        let generator = ArithmeticGenerator::<L>::new(trace_data);
        let writer = generator.new_writer();
        for i in 0..L::num_rows() {
            let a = CubicElement([GoldilocksField::rand(); 3]);
            let b = CubicElement([GoldilocksField::rand(); 3]);
            writer.write(&x_1, &a, L::num_rows() - 1 - i);
            writer.write(&x_2, &a, i);
            writer.write(&x_3, &b, i);
            writer.write(&x_4, &b, i);
        }

        let stark = Starky::from_chip(air);

        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
