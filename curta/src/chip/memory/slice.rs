use crate::chip::register::cubic::CubicRegister;

use super::{value::MemoryValue, pointer::RawPointer};




pub struct SlicePointer {
    ptr: RawPointer,
    idx_challenge : CubicRegister,
    len: usize,
}