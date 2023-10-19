use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::memory::pointer::Pointer;
use crate::chip::memory::time::Time;
use crate::chip::memory::value::MemoryValue;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::Register;
use crate::chip::AirParameters;
use crate::math::field::PrimeField64;

/// A safe interface for an AIR builder.
pub trait Builder {
    type Field: PrimeField64;
    type Parameters: AirParameters<Field = Self::Field>;

    /// Returns the underlying AIR builder.
    fn api(&mut self) -> &mut AirBuilder<Self::Parameters>;

    /// Allocates a trace register.
    fn alloc<T: Register>(&mut self) -> T {
        self.api().alloc()
    }

    /// Allocates a register in public inputs.
    fn alloc_public<T: Register>(&mut self) -> T {
        self.api().alloc_public()
    }

    /// Allocates a constant register with set value `value`.
    fn constant<T: Register>(&mut self, value: &T::Value<Self::Field>) -> T {
        self.api().constant(value)
    }

    /// Initializes a pointer with initial `value` and write time given by `time`.
    fn initialize<V: MemoryValue>(
        &mut self,
        value: &V,
        time: &Time<Self::Field>,
        multiplicity: Option<ElementRegister>,
    ) -> Pointer<V> {
        self.api().initialize(value, time, multiplicity)
    }

    /// Reads the memory at location `ptr` with last write time given by `last_write_ts`.
    fn get<V: MemoryValue>(&mut self, ptr: &Pointer<V>, last_write_ts: &Time<Self::Field>) -> V {
        self.api().get(ptr, last_write_ts)
    }

    /// Writes `value` to the memory at location `ptr` with write time given by `write_ts`. Values
    /// can be written with an optional `multiplicity`.
    ///
    /// If `multiplicity` is `None`, then the value is written to the memory bus with multiplicity
    /// set to 1 allowing a single read. If `multiplicity` is `Some(m)`, then the value is written
    /// to the memory bus with multiplicity given by the value of `m`, allowing `m` reads.
    fn set<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        value: V,
        write_ts: &Time<Self::Field>,
        multiplicity: Option<ElementRegister>,
    ) {
        self.api().set(ptr, value, write_ts, multiplicity)
    }

    /// Frees the memory at location `ptr` with last write time given by `last_write_ts`.
    fn free<V: MemoryValue>(&mut self, ptr: &Pointer<V>, value: V, last_write: &Time<Self::Field>) {
        self.api().free(ptr, value, last_write)
    }

    /// Asserts that `a = b` in all rows of the trace.
    fn assert_equal<T: Register>(&mut self, a: &T, b: &T) {
        self.api().assert_equal(a, b)
    }

    /// Asserts that `a = b` in the first row of the trace.
    fn assert_equal_first_row<T: Register>(&mut self, a: &T, b: &T) {
        self.api().assert_equal_first_row(a, b)
    }

    /// Asserts that `a = b` in the last row of the trace.
    fn assert_equal_last_row<T: Register>(&mut self, a: &T, b: &T) {
        self.api().assert_equal_last_row(a, b)
    }

    /// Asserts that `a = b` in the transition constraints of the trace.
    fn assert_equal_transition<T: Register>(&mut self, a: &T, b: &T) {
        self.api().assert_equal_transition(a, b)
    }

    /// Asserts that `expression = 0` in all rows of the trace.
    fn assert_expression_zero(&mut self, expression: ArithmeticExpression<Self::Field>) {
        self.api().assert_expression_zero(expression)
    }

    /// Asserts that `expression = 0` in the first row of the trace.
    fn assert_expression_zero_first_row(&mut self, expression: ArithmeticExpression<Self::Field>) {
        self.api().assert_expression_zero_first_row(expression)
    }

    /// Asserts that `expression = 0` in the last row of the trace.
    fn assert_expression_zero_last_row(&mut self, expression: ArithmeticExpression<Self::Field>) {
        self.api().assert_expression_zero_last_row(expression)
    }

    /// Asserts that `expression = 0` in the transition constraints of the trace.
    fn assert_expression_zero_transition(&mut self, expression: ArithmeticExpression<Self::Field>) {
        self.api().assert_expression_zero_transition(expression)
    }

    fn assert_expressions_equal(
        &mut self,
        a: ArithmeticExpression<Self::Field>,
        b: ArithmeticExpression<Self::Field>,
    ) {
        self.api().assert_expressions_equal(a, b)
    }

    /// Computes the expression `a + b` and sets the result to register `value`.
    fn set_to_expression<T: Register>(
        &mut self,
        register: &T,
        expression: ArithmeticExpression<Self::Field>,
    ) {
        // if expression.is_trace() {
        //     assert!(
        //         register.is_trace(),
        //         "Cannot set a non-trace register to a trace expression"
        //     );
        //     self.api().set_to_expression(register, expression)
        // }
        match (register.is_trace(), expression.is_trace()) {
            (true, _) => self.api().set_to_expression(register, expression),
            (false, true) => panic!("Cannot set a non-trace register to a trace expression"),
            (false, false) => self.api().set_to_expression_public(register, expression),
        }
    }

    fn set_to_expression_first_row<T: Register>(
        &mut self,
        register: &T,
        expression: ArithmeticExpression<Self::Field>,
    ) {
        self.api().set_to_expression_first_row(register, expression);
    }

    fn set_to_expression_last_row<T: Register>(
        &mut self,
        register: &T,
        expression: ArithmeticExpression<Self::Field>,
    ) {
        self.api().set_to_expression_last_row(register, expression);
    }

    fn set_to_expression_transition<T: Register>(
        &mut self,
        register: &T,
        expression: ArithmeticExpression<Self::Field>,
    ) {
        assert!(
            matches!(register.register(), MemorySlice::Next(_, _)),
            "Cannot set a current register in a transition constraint"
        );
        self.api()
            .set_to_expression_transition(register, expression);
    }
}

impl<L: AirParameters> Builder for AirBuilder<L> {
    type Field = L::Field;
    type Parameters = L;

    fn api(&mut self) -> &mut AirBuilder<L> {
        self
    }
}
