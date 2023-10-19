use super::Builder;

/// The addition operation.
///
/// Types implementing this trait can be used within the `builder.add(lhs, rhs)` method.
pub trait Add<B: Builder, Rhs = Self> {
    type Output;

    fn add(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The subtraction operation.
///
/// Types implementing this trait can be used within the `builder.sub(lhs, rhs)` method.
pub trait Sub<B: Builder, Rhs = Self> {
    type Output;

    fn sub(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The multiplication operation.
///
/// Types implementing this trait can be used within the `builder.mul(lhs, rhs)` method.
pub trait Mul<B: Builder, Rhs = Self> {
    type Output;

    fn mul(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The negation operation.
///
/// Types implementing this trait can be used within the `builder.neg(value)` method.
pub trait Neg<B: Builder> {
    type Output;

    fn neg(self, builder: &mut B) -> Self::Output;
}

/// A zero element.
///
/// Types implementing this trait can be used via the `builder.zero()` method.
pub trait Zero<B: Builder> {
    type Output;

    fn zero(builder: &mut B) -> Self::Output;
}

/// The bitwise AND operation.
///
/// Types implementing this trait can be used within the `builder.and(lhs, rhs)` method.
pub trait And<B: Builder, Rhs = Self> {
    type Output;

    fn and(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The bitwise OR operation.
///
/// Types implementing this trait can be used within the `builder.or(lhs, rhs)` method.
pub trait Or<B: Builder, Rhs = Self> {
    type Output;

    fn or(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The bitwise XOR operation.
///
/// Types implementing this trait can be used within the `builder.xor(lhs, rhs)` method.
pub trait Xor<B: Builder, Rhs = Self> {
    type Output;

    fn xor(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The left shift operation.
///
/// Types implementing this trait can be used within the `builder.shl(lhs, rhs)` method.
pub trait Shl<B: Builder, Rhs = Self> {
    type Output;

    fn shl(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The right shift operation.
///
/// Types implementing this trait can be used within the `builder.shr(lhs, rhs)` method.
pub trait Shr<B: Builder, Rhs = Self> {
    type Output;

    fn shr(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The rotate left operation.
///
/// Types implementing this trait can be used within the `builder.rotate_left(lhs, rhs)` method.
pub trait RotateLeft<B: Builder, Rhs = Self> {
    type Output;

    fn rotate_left(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}

/// The rotate right operation.
///
/// Types implementing this trait can be used within the `builder.rotate_right(lhs, rhs)` method.
pub trait RotateRight<B: Builder, Rhs = Self> {
    type Output;

    fn rotate_right(self, rhs: Rhs, builder: &mut B) -> Self::Output;
}
