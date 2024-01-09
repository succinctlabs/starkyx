use core::fmt::Debug;

use num::Num;

use super::builder::Builder;
use crate::chip::memory::value::MemoryValue;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::Register;

pub mod blake;
pub mod sha;

pub trait HashPureInteger {
    type Integer: Num + Copy + Debug;
}

pub trait HashInteger<B: Builder> {
    type IntRegister: MemoryValue + Register<Value<B::Field> = Self::Value>;
    type Value;
}

pub trait HashIntConversion<B: Builder>: HashInteger<B> + HashPureInteger {
    /// Convert an integer to the `Self::IntRegister` field value.
    fn int_to_field_value(int: Self::Integer) -> Self::Value;

    /// Convert a `Self::IntRegister` field value to an integer.
    fn field_value_to_int(value: &Self::Value) -> Self::Integer;
}

pub trait HashDigest<B: Builder>: HashInteger<B> {
    type DigestRegister: Register + Into<ArrayRegister<Self::IntRegister>>;
}
