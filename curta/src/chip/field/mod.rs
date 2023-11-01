//! Implements field arithmetic for any field, using a technique from Polygon Zero.
//! Reference: https://github.com/mir-protocol/plonky2/blob/main/evm/src/arithmetic/addcy.rs
//!
//! We want to compute a + b = result mod p. In the integers, this is equivalent to witnessing some
//! carry such that:
//!
//! a + b - result - carry * p = 0.
//!
//! Let us encode the integers as polynomials in the Goldilocks field, where each coefficient is
//! at most 16 bits. In other words, the integers are encoded as an array of little-endian base 16
//! limbs. We can then write the above equation as:
//!
//! a(x) + b(x) - result(x) - carry(x) * p(x)
//!
//! where the polynomial should evaluate to 0 if x = 2^16. To prove that the polynomial has a root
//! at 2^16, we can have the prover witness a polynomial `w(x)` such that the above polynomial
//! is divisble by (x - 2^16):
//!
//! a(x) + b(x) - result(x) - carry(x) * p(x) - (x - 2^16) * w(x) = 0
//!
//! Thus, if we can prove that above polynomial is 0, we can conclude that the addition has been
//! computed correctly. Note that this relies on the fact that any quadratic sum of a sufficiently
//! small number of terms (i.e., less than 2^32 terms) will not overflow in the Goldilocks field.
//! Furthermore, one must be careful to ensure that all polynomials except w(x) are range checked
//! in [0, 2^16).
//!
//! This technique generalizes for any quadratic sum with a "small" number of terms to avoid
//! overflow.

pub mod add;
pub mod constants;
pub mod den;
pub mod div;
pub mod inner_product;
pub mod instruction;
pub mod mul;
pub mod mul_const;
pub mod ops;
pub mod parameters;
pub mod register;
pub mod sub;
mod util;
