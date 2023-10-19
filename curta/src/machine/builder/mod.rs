use self::ops::{Adc, Add, And, Mul, Neg, One, Or, Shl, Shr, Sub, Xor, Zero};
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::pointer::Pointer;
use crate::chip::memory::time::Time;
use crate::chip::memory::value::MemoryValue;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::slice::RegisterSlice;
use crate::chip::register::Register;
use crate::chip::AirParameters;
use crate::math::field::PrimeField64;

pub mod ops;

/// A safe interface for an AIR builder.
pub trait Builder: Sized {
    type Field: PrimeField64;
    type Parameters: AirParameters<Field = Self::Field>;

    /// Returns the underlying AIR builder.
    fn api(&mut self) -> &mut AirBuilder<Self::Parameters>;

    /// Allocates a trace register.
    fn alloc<T: Register>(&mut self) -> T {
        self.api().alloc()
    }

    /// Allocates a trace array register with type `T` and length `len`.
    fn alloc_array<T: Register>(&mut self, len: usize) -> ArrayRegister<T> {
        self.api().alloc_array(len)
    }

    /// Allocates a register in public inputs.
    fn alloc_public<T: Register>(&mut self) -> T {
        self.api().alloc_public()
    }

    /// Allocates an array register in public inputs with type `T` and length `len`.
    fn alloc_array_public<T: Register>(&mut self, len: usize) -> ArrayRegister<T> {
        self.api().alloc_array_public(len)
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

    /// Initializes a slice of mutable memory with initial `value` and write time given by `time`.
    fn initialize_slice<V: MemoryValue>(
        &mut self,
        values: &impl RegisterSlice<V>,
        time: &Time<Self::Field>,
        multiplicity: Option<ElementRegister>,
    ) -> Slice<V> {
        self.api().initialize_slice(values, time, multiplicity)
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

    fn add<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Add<Self, Rhs>>::Output
    where
        Lhs: Add<Self, Rhs>,
    {
        lhs.add(rhs, self)
    }

    fn sub<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Sub<Self, Rhs>>::Output
    where
        Lhs: Sub<Self, Rhs>,
    {
        lhs.sub(rhs, self)
    }

    fn mul<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Mul<Self, Rhs>>::Output
    where
        Lhs: Mul<Self, Rhs>,
    {
        lhs.mul(rhs, self)
    }

    fn zero<T: Zero<Self>>(&mut self) -> <T as Zero<Self>>::Output {
        T::zero(self)
    }

    fn one<T: One<Self>>(&mut self) -> <T as One<Self>>::Output {
        T::one(self)
    }

    fn neg<T: Neg<Self>>(&mut self, value: T) -> <T as Neg<Self>>::Output {
        value.neg(self)
    }

    fn carrying_add<Lhs, Rhs, Carry>(
        &mut self,
        lhs: Lhs,
        rhs: Rhs,
        carry: Carry,
    ) -> <Lhs as ops::Adc<Self, Rhs, Carry>>::Output
    where
        Lhs: Adc<Self, Rhs, Carry>,
    {
        lhs.adc(rhs, carry, self)
    }

    fn and<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::And<Self, Rhs>>::Output
    where
        Lhs: And<Self, Rhs>,
    {
        lhs.and(rhs, self)
    }

    fn or<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Or<Self, Rhs>>::Output
    where
        Lhs: Or<Self, Rhs>,
    {
        lhs.or(rhs, self)
    }

    fn xor<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Xor<Self, Rhs>>::Output
    where
        Lhs: Xor<Self, Rhs>,
    {
        lhs.xor(rhs, self)
    }

    fn shl<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Shl<Self, Rhs>>::Output
    where
        Lhs: Shl<Self, Rhs>,
    {
        lhs.shl(rhs, self)
    }

    fn shr<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Shr<Self, Rhs>>::Output
    where
        Lhs: Shr<Self, Rhs>,
    {
        lhs.shr(rhs, self)
    }

    fn rotate_left<Lhs, Rhs>(
        &mut self,
        lhs: Lhs,
        rhs: Rhs,
    ) -> <Lhs as ops::RotateLeft<Self, Rhs>>::Output
    where
        Lhs: ops::RotateLeft<Self, Rhs>,
    {
        lhs.rotate_left(rhs, self)
    }

    fn rotate_right<Lhs, Rhs>(
        &mut self,
        lhs: Lhs,
        rhs: Rhs,
    ) -> <Lhs as ops::RotateRight<Self, Rhs>>::Output
    where
        Lhs: ops::RotateRight<Self, Rhs>,
    {
        lhs.rotate_right(rhs, self)
    }
}

impl<L: AirParameters> Builder for AirBuilder<L> {
    type Field = L::Field;
    type Parameters = L;

    fn api(&mut self) -> &mut AirBuilder<L> {
        self
    }
}
