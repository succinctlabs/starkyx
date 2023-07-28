use super::entry::Entry;
use super::BusChannel;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;
use crate::chip::register::{Register, RegisterSerializable};
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::math::prelude::*;
use crate::plonky2::field::cubic::element::CubicElement;

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP>
    for BusChannel<AP::Field, E>
{
    fn eval(&self, parser: &mut AP) {
        let beta = self.challenge.eval(parser);

        // Constrain the entry_values
        for (entry, value) in self.entries.iter().zip(self.entry_values.iter()) {
            let val = value.eval(parser);
            entry.eval_entry(beta, val, parser);
        }

        // Constrain the row-wise running product values
        let product_chunck = self.entry_values.chunks_exact(2);

        // save the last element for bus constraint
        let last_element = product_chunck.remainder().first();

        // Constraint the running product, picking the values for the first row
        let mut prev = parser.one_extension();
        let mut prev_first = parser.one_extension();
        for (chunk, acc_reg) in product_chunck.zip(self.row_acc_product.iter()) {
            let a = chunk[0].next().eval(parser);
            let b = chunk[1].next().eval(parser);
            let acc = acc_reg.next().eval(parser);

            let a_first = chunk[0].eval(parser);
            let b_first = chunk[1].eval(parser);
            let acc_first = acc_reg.eval(parser);

            let mut acc_constraint = parser.mul_extension(prev, a);
            acc_constraint = parser.mul_extension(acc_constraint, b);
            acc_constraint = parser.sub_extension(acc, acc_constraint);
            parser.constraint_extension(acc_constraint);
            prev = acc;

            let mut acc_first_constraint = parser.mul_extension(prev_first, a_first);
            acc_first_constraint = parser.mul_extension(acc_first_constraint, b_first);
            acc_first_constraint = parser.sub_extension(acc_first, acc_first_constraint);
            parser.constraint_extension(acc_first_constraint);
            prev_first = acc_first;
        }

        // Constraint the bus values
        let mut prod_value = prev;
        let mut prod_value_first = prev_first;
        if let Some(last_element) = last_element {
            prod_value = parser.mul_extension(prod_value, last_element.next().eval(parser));
            prod_value_first = parser.mul_extension(prod_value_first, last_element.eval(parser));
        }

        let bus_value = self.table_accumulator.eval(parser);
        let bus_next_value = self.table_accumulator.next().eval(parser);

        let bus_first_row_consr = parser.sub_extension(bus_value, prod_value_first);
        parser.constraint_extension_first_row(bus_first_row_consr);

        let bus_next_expected = parser.mul_extension(bus_value, prod_value);
        let bus_next_consr = parser.sub_extension(bus_next_value, bus_next_expected);
        parser.constraint_extension_transition(bus_next_consr);

        // Constraint the out channel to the last row of the bus column
        let out_channel = self.out_channel.eval(parser);
        let out_minus_bus = parser.sub_extension(out_channel, bus_value);
        parser.constraint_extension_last_row(out_minus_bus);
    }
}

impl<F: Field> Entry<F> {
    pub fn eval_entry<E: CubicParameters<F>, AP: CubicParser<E>>(
        &self,
        beta: CubicElement<AP::Var>,
        entry_value: CubicElement<AP::Var>,
        parser: &mut AP,
    ) where
        AP: AirParser<Field = F>,
    {
        match self {
            Entry::Input(value, filter) => {
                let filter_vec = filter.eval(parser);
                assert_eq!(filter_vec.len(), 1);
                let one = parser.one_extension();
                let filter = parser.element_from_base_field(filter_vec[0]);

                let mut expected_value = value.eval(parser);
                expected_value = parser.sub_extension(beta, expected_value);
                expected_value = parser.mul_extension(filter, expected_value);

                let not_filter = parser.sub_extension(one, filter);
                expected_value = parser.add_extension(not_filter, expected_value);

                let constraint = parser.sub_extension(entry_value, expected_value);
                parser.constraint_extension(constraint);
            }
            Entry::Output(value, filter) => {
                let filter_vec = filter.eval(parser);
                let one = parser.one_extension();
                let filter = parser.element_from_base_field(filter_vec[0]);
                let not_filter = parser.sub_extension(one, filter);

                let mut inverse_value = value.eval(parser);
                inverse_value = parser.sub_extension(beta, inverse_value);
                inverse_value = parser.mul_extension(filter, inverse_value);

                // Check the value is an inverse of the expected value if filter = one
                let product = parser.mul_extension(entry_value, inverse_value);
                let inverse_constraint = parser.sub_extension(filter, product);
                parser.constraint_extension(inverse_constraint);

                // Check that entry_value is assigned correctly
                let not_value = parser.mul_extension(not_filter, one);
                let mut not_value_constraint = parser.mul_extension(not_filter, entry_value);
                not_value_constraint = parser.sub_extension(not_value, not_value_constraint);
                parser.constraint_extension(not_value_constraint);
            }
        }
    }
}
