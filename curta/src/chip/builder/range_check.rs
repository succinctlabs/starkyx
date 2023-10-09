use super::AirBuilder;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::table::lookup::table::LookupTable;
use crate::chip::table::lookup::values::LookupValues;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    pub(crate) fn arithmetic_range_checks(&mut self) {
        let table = self.alloc::<ElementRegister>();

        let one = || -> ArithmeticExpression<L::Field> { ArithmeticExpression::one() };
        let zero = || -> ArithmeticExpression<L::Field> { ArithmeticExpression::zero() };

        // Table constraints
        self.assert_expressions_equal_first_row(table.expr(), zero());
        self.assert_expressions_equal_transition(table.expr() + one(), table.next().expr());

        let values = ArrayRegister::<ElementRegister>::from_register_unsafe(MemorySlice::Local(
            0,
            L::NUM_ARITHMETIC_COLUMNS,
        ))
        .into_iter()
        .chain(self.global_arithmetic.iter().copied())
        .collect::<Vec<_>>();

        let multiplicities = self.alloc_array::<ElementRegister>(1);
        let mut table_data = self.new_lookup(&[table], &multiplicities);
        let lookup_values = table_data.register_lookup_values(self, &values);
        self.constrain_element_lookup_table(table_data.clone());

        self.range_data = Some((
            LookupTable::Element(table_data),
            LookupValues::Element(lookup_values),
        ));
    }
}
