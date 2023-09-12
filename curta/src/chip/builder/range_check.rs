use super::AirBuilder;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    pub(crate) fn arithmetic_range_checks(&mut self) {
        let table = self.alloc::<ElementRegister>();

        self.range_table = Some(table);

        let one = || -> ArithmeticExpression<L::Field> { ArithmeticExpression::one() };
        let zero = || -> ArithmeticExpression<L::Field> { ArithmeticExpression::zero() };

        // Table constraints
        self.assert_expressions_equal_first_row(table.expr(), zero());
        self.assert_expressions_equal_transition(table.expr() + one(), table.next().expr());

        assert_eq!(
            L::NUM_ARITHMETIC_COLUMNS % 2,
            0,
            "The number of arithmetic columns must be even"
        );
        let values = ArrayRegister::<ElementRegister>::from_register_unsafe(MemorySlice::Local(
            0,
            L::NUM_ARITHMETIC_COLUMNS,
        ))
        .into_iter()
        .chain(self.global_arithmetic.iter().copied())
        .collect::<Vec<_>>();

        self.element_lookup_with_table_index(&table, &values, Self::range_fn)
    }
}
