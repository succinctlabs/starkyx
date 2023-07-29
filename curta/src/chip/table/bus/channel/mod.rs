pub mod constraint;
pub mod entry;
pub mod trace;

use core::marker::PhantomData;

use self::entry::Entry;
use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[derive(Debug, Clone)]
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
    pub fn input_to_bus(
        &mut self,
        bus: &mut BusChannel<L::Field, L::CubicParams>,
        register: CubicRegister,
    ) {
        bus.input(register);
        bus.entry_values.push(self.alloc_extended());
        if bus.entries.len() % 2 == 0 {
            let product = self.alloc_extended::<CubicRegister>();
            bus.row_acc_product.push(product);
        }
    }

    pub fn input_to_bus_filtered(
        &mut self,
        bus: &mut BusChannel<L::Field, L::CubicParams>,
        register: CubicRegister,
        filter: ArithmeticExpression<L::Field>,
    ) {
        bus.input_filtered(register, filter);
        bus.entry_values.push(self.alloc_extended());
        if bus.entries.len() % 2 == 0 {
            let product = self.alloc_extended::<CubicRegister>();
            bus.row_acc_product.push(product);
        }
    }

    pub fn output_from_bus(
        &mut self,
        bus: &mut BusChannel<L::Field, L::CubicParams>,
        register: CubicRegister,
    ) {
        bus.output(register);
        bus.entry_values.push(self.alloc_extended());
        if bus.entries.len() % 2 == 0 {
            let product = self.alloc_extended::<CubicRegister>();
            bus.row_acc_product.push(product);
        }
    }

    pub fn output_from_bus_filtered(
        &mut self,
        bus: &mut BusChannel<L::Field, L::CubicParams>,
        register: CubicRegister,
        filter: ArithmeticExpression<L::Field>,
    ) {
        bus.output_filtered(register, filter);
        bus.entry_values.push(self.alloc_extended());
        if bus.entries.len() % 2 == 0 {
            let product = self.alloc_extended::<CubicRegister>();
            bus.row_acc_product.push(product);
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
    use crate::chip::builder::tests::*;
    use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
    use crate::chip::register::Register;
    use crate::chip::AirParameters;
    use crate::math::extension::cubic::element::CubicElement;

    #[derive(Debug, Clone)]
    struct BusChannelTest;

    impl AirParameters for BusChannelTest {
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
    fn test_bus_channel() {
        type L = BusChannelTest;
        type F = GoldilocksField;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();
        let x_1 = builder.alloc::<CubicRegister>();
        let x_2 = builder.alloc::<CubicRegister>();

        let beta = builder.alloc_challenge::<CubicRegister>();
        let out_channel = builder.alloc_global::<CubicRegister>();
        let accumulator = builder.alloc_extended::<CubicRegister>();
        let mut channel = BusChannel::new(beta, out_channel, accumulator);
        builder.bus_channels.push(channel.clone());

        builder.input_to_bus(&mut channel, x_1);
        builder.output_from_bus(&mut channel, x_2);

        let one = ArithmeticExpression::<F>::one();
        let zero = ArithmeticExpression::<F>::zero();

        let [c_0, c_1, c_2] = out_channel.as_base_array();
        builder.assert_expressions_equal(c_0.expr(), one);
        builder.assert_expressions_equal(c_1.expr(), zero.clone());
        builder.assert_expressions_equal(c_2.expr(), zero);

        let air = builder.build();

        let generator = ArithmeticGenerator::<L>::new_from_air(&air);
        let writer = generator.new_writer();
        for i in 0..L::num_rows() {
            let a = CubicElement([GoldilocksField::rand(); 3]);
            writer.write(&x_1, &a, i);
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
