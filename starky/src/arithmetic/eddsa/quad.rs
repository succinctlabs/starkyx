use crate::arithmetic::{ArithmeticOp, ArithmeticParser, Register};

/// A gadget to compute
/// QUAD(x, y, z, w) = (a * b + c * d) mod p
#[derive(Debug, Clone, Copy)]
pub struct QuadLayout {
    a: Register,
    b: Register,
    c: Register,
    d: Register,
    output: Register,
    witness_low: Register,
    witness_high: Register,
}
