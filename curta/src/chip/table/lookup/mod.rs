//! This module implements a lookup argument based on the logarithmic derivative as in
//! https://eprint.iacr.org/2022/1530.pdf
//!

use self::table::LogLookupTable;
use self::values::LogLookupValues;

pub mod constraint;
pub mod table;
pub mod trace;
pub mod values;
