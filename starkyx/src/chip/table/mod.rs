//! Table level constraints for inter-column claims.
//!
//! Currently, we have three types of claims:
//!
//!  - Lookup: A column is a lookup table for another column. Namely, a lookup argument allows for
//!       a proof that all the values of columns `a_1`, ..., `a_n` are contained in the entries
//!       of column `b` (the table).
//!
//! - Permutation: A column is a permutation of another column.
//!
//! - Evaluation: The values of a column (or a subset of its rows) is the same as that
//!                 of another column (or the same subset of its rows).
//!
//!

pub mod accumulator;
pub mod bus;
pub mod log_derivative;
pub mod lookup;
pub mod powers;
