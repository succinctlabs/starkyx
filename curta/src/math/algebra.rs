use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use super::field::{Field, Ring};

pub trait Algebra<F: Field>:
    Ring + From<F> + Add<F, Output = Self> + Sub<F, Output = Self> + Mul<F, Output = Self>
{
}

impl<F: Field> Algebra<F> for F {}
