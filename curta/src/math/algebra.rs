use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use super::field::{Field, Ring};

pub trait Algebra<F: Field>:
    Ring
    + From<F>
    + Add<F, Output = Self>
    + AddAssign<F>
    + Sub<F, Output = Self>
    + SubAssign<F>
    + Mul<F, Output = Self>
    + MulAssign<F>
{
}

impl<F: Field> Algebra<F> for F {}
