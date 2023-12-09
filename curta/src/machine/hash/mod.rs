use core::fmt::Debug;

use num::Num;

use super::builder::Builder;
use crate::chip::memory::value::MemoryValue;
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
