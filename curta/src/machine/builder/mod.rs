use self::ops::{Adc, Add, And, Div, Double, Mul, Neg, Not, One, Or, Shl, Shr, Sub, Xor, Zero};
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::ec::scalar::LimbBitInstruction;
use crate::chip::instruction::cycle::Cycle;
use crate::chip::instruction::Instruction;
use crate::chip::memory::instruction::MemorySliceIndex;
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::memory::pointer::Pointer;
use crate::chip::memory::time::Time;
use crate::chip::memory::value::MemoryValue;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::memory::MemorySlice;
use crate::chip::register::slice::RegisterSlice;
use crate::chip::register::Register;
use crate::chip::AirParameters;
use crate::math::field::PrimeField64;
use crate::math::prelude::CubicParameters;

pub mod ops;

/// A safe interface for an AIR builder.
pub trait Builder: Sized {
    type Field: PrimeField64;
    type CubicParams: CubicParameters<Self::Field>;
    type Instruction: Instruction<Self::Field>;
    type Parameters: AirParameters<
        Field = Self::Field,
        CubicParams = Self::CubicParams,
        Instruction = Self::Instruction,
    >;

    /// Returns the underlying AIR builder.
    fn api(&mut self) -> &mut AirBuilder<Self::Parameters>;

    fn clk(&mut self) -> ElementRegister;

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

    fn constant_array<T: Register>(
        &mut self,
        values: &[T::Value<Self::Field>],
    ) -> ArrayRegister<T> {
        self.api().constant_array(values)
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

    /// Creates a pointer which is not initialized to any value.
    fn uninit<V: MemoryValue>(&mut self) -> Pointer<V> {
        self.api().uninit()
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

    /// Creates an uninitialized slice reference.
    fn uninit_slice<V: MemoryValue>(&mut self) -> Slice<V> {
        self.api().uninit_slice()
    }

    /// Reads the memory at location `ptr` with last write time given by `last_write_ts`.
    fn load<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        last_write_ts: &Time<Self::Field>,
        label: Option<String>,
        index: Option<MemorySliceIndex>,
    ) -> V {
        self.api().get(ptr, last_write_ts, label, index)
    }

    /// Writes `value` to the memory at location `ptr` with write time given by `write_ts`. Values
    /// can be written with an optional `multiplicity`.
    ///
    /// If `multiplicity` is `None`, then the value is written to the memory bus with multiplicity
    /// set to 1 allowing a single read. If `multiplicity` is `Some(m)`, then the value is written
    /// to the memory bus with multiplicity given by the value of `m`, allowing `m` reads.
    fn store<V: MemoryValue>(
        &mut self,
        ptr: &Pointer<V>,
        value: V,
        write_ts: &Time<Self::Field>,
        multiplicity: Option<ElementRegister>,
        label: Option<String>,
        index: Option<MemorySliceIndex>,
    ) {
        self.api()
            .set(ptr, value, write_ts, multiplicity, label, index)
    }
    /// Frees the memory at location `ptr` with last write time given by `last_write_ts`.
    fn free<V: MemoryValue>(&mut self, ptr: &Pointer<V>, value: V, last_write: &Time<Self::Field>) {
        self.api().free(ptr, value, last_write)
    }

    /// Prints out a log message (using the log::debug! macro) with the value and multiplicity
    /// of the memory slot.
    ///
    /// The message will be presented with `RUST_LOG=debug` or `RUST_LOG=trace`.
    fn watch_memory<V: MemoryValue>(&mut self, ptr: &Pointer<V>, name: &str) {
        self.api().watch_memory(ptr, name)
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

    /// Sets `dest` in the next row of the trace to equal the value of `src` in the current row.
    fn set_next<T: Register>(&mut self, dest: &T, src: &T) {
        self.set_to_expression_transition(&dest.next(), src.expr());
    }

    /// Sets `dest` in the first row of the trace to equal the value of `src` in the current row.
    fn set_first_row<T: Register>(&mut self, dest: &T, src: &T) {
        assert!(
            matches!(dest.register(), MemorySlice::Local(_, _)),
            "Cannot set non-local register in a first row constraint"
        );
        self.set_to_expression_first_row(dest, src.expr());
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
        match (register.is_trace(), expression.is_trace()) {
            (true, _) => self.api().set_to_expression(register, expression),
            (false, true) => panic!("Cannot set a non-trace register to a trace expression"),
            (false, false) => self.api().set_to_expression_public(register, expression),
        }
    }

    fn select<T: Register>(&mut self, flag: BitRegister, true_value: &T, false_value: &T) -> T {
        self.api().select(&flag, true_value, false_value)
    }

    fn select_next<T: Register>(
        &mut self,
        flag: BitRegister,
        true_value: &T,
        false_value: &T,
        result: &T,
    ) {
        self.set_to_expression_transition(
            &result.next(),
            flag.expr() * true_value.expr() + flag.not_expr() * false_value.expr(),
        );
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

    fn set_next_expression<T: Register>(
        &mut self,
        dest: &T,
        src: ArithmeticExpression<Self::Field>,
    ) {
        assert!(
            matches!(dest.register(), MemorySlice::Local(_, _)),
            "Cannot set a next register in a transition constraint"
        );
        self.api().set_to_expression_transition(&dest.next(), src);
    }

    /// Computes the expression `expression` and returns the result as a trace register of type `T`.
    fn expression<T: Register>(&mut self, expression: ArithmeticExpression<Self::Field>) -> T {
        let register = self.alloc::<T>();
        self.set_to_expression(&register, expression);
        register
    }

    /// Prints out a log message (using the log::debug! macro) with the value of the register.
    ///
    /// The message will be presented with `RUST_LOG=debug` or `RUST_LOG=trace`.
    fn watch(&mut self, data: &impl Register, name: &str) {
        self.api().watch(data, name);
    }

    /// Computes the expression `expression` and returns the result as a public register of type `T`.
    fn public_expression<T: Register>(
        &mut self,
        expression: ArithmeticExpression<Self::Field>,
    ) -> T {
        let register = self.alloc_public::<T>();
        self.set_to_expression(&register, expression);
        register
    }

    fn add<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Add<Self, Rhs>>::Output
    where
        Lhs: Add<Self, Rhs>,
    {
        lhs.add(rhs, self)
    }

    fn double<T: Double<Self>>(&mut self, value: T) -> <T as Double<Self>>::Output {
        value.double(self)
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

    fn div<Lhs, Rhs>(&mut self, lhs: Lhs, rhs: Rhs) -> <Lhs as ops::Div<Self, Rhs>>::Output
    where
        Lhs: Div<Self, Rhs>,
    {
        lhs.div(rhs, self)
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

    fn not<T: Not<Self>>(&mut self, value: T) -> <T as Not<Self>>::Output {
        value.not(self)
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

    fn cycle(&mut self, length_log: usize) -> Cycle<Self::Field> {
        self.api().cycle(length_log)
    }

    /// `process_id` is a register is computed by counting the number of cycles. We do this by
    /// setting `process_id` to be the cumulative sum of the `end_bit` of each cycle.
    fn process_id(&mut self, size: usize, end_bit: BitRegister) -> ElementRegister {
        self.api().process_id(size, end_bit)
    }

    fn bit_decomposition(
        &mut self,
        limb: ElementRegister,
        start_bit: BitRegister,
        end_bit: BitRegister,
    ) -> BitRegister
    where
        Self::Instruction: From<LimbBitInstruction>,
    {
        self.api().bit_decomposition(limb, start_bit, end_bit)
    }
}

impl<L: AirParameters> Builder for AirBuilder<L> {
    type Field = L::Field;
    type CubicParams = L::CubicParams;
    type Parameters = L;
    type Instruction = L::Instruction;

    fn api(&mut self) -> &mut AirBuilder<L> {
        self
    }

    fn clk(&mut self) -> ElementRegister {
        self.clock()
    }
}
