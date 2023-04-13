use core::fmt::Debug;

use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

/// A gadget used for polynomial operations inside Plonky2. Useful for writing constraint verifiers
/// in the extension field for verifying STARKs.
#[derive(Debug, Clone, Copy)]
pub struct PolynomialGadget;

impl PolynomialGadget {
    /// Creates a constant extension field target in the circuit from an extension field target
    /// (vectorized).
    pub fn constant_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[F::Extension],
    ) -> Vec<ExtensionTarget<D>> {
        a.iter().map(|x| builder.constant_extension(*x)).collect()
    }

    /// Adds two extension field targets in the circuit and outputs the result (vectorized).
    pub fn add_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &[ExtensionTarget<D>],
    ) -> Vec<ExtensionTarget<D>> {
        a.iter()
            .zip_longest(b.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => builder.add_extension(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => *b,
            })
            .collect()
    }

    /// Subtracts two extension field targets in the circuit and outputs the result (vectorized).
    pub fn sub_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &[ExtensionTarget<D>],
    ) -> Vec<ExtensionTarget<D>> {
        let zero = builder.constant_extension(F::Extension::ZERO);
        a.iter()
            .zip_longest(b.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => builder.sub_extension(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => builder.sub_extension(zero, *b),
            })
            .collect()
    }

    /// Subtracts a constant extension field target from an existing extension field target
    /// in the circuit and outputs the result (vectorized).
    pub fn sub_constant_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &ExtensionTarget<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let len = a.len();

        let mut result = Vec::with_capacity(len);
        for elem in a {
            result.push(builder.sub_extension(*elem, *b));
        }
        result
    }

    /// Multiplies two extension field targets in the circuit and outputs the result (vectorized).
    pub fn mul_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &[ExtensionTarget<D>],
    ) -> Vec<ExtensionTarget<D>> {
        let mut result = vec![builder.constant_extension(F::ZERO.into()); a.len() + b.len() - 1];

        for i in 0..a.len() {
            for j in 0..b.len() {
                let aibj = builder.mul_extension(a[i], b[j]);
                result[i + j] = builder.add_extension(result[i + j], aibj);
            }
        }

        result
    }

    /// Scalar multiplication of an extension field target by a scalar in the circuit and outputs
    /// the result in the circuit (vectorized).
    pub fn scalar_mul_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &Target,
    ) -> Vec<ExtensionTarget<D>> {
        a.iter().map(|x| builder.scalar_mul_ext(*b, *x)).collect()
    }

    /// Scalar multiplication of an extension field target by an extension field scalar in the
    /// circuit and outputs the result in the circuit (vectorized).
    pub fn ext_scalar_mul_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &ExtensionTarget<D>,
    ) -> Vec<ExtensionTarget<D>> {
        a.iter().map(|x| builder.mul_extension(*b, *x)).collect()
    }

    /// Scalar multiplication of two polynomials in the circuit where one polynomial is in the
    /// extension field and one is in the native field. Outputs the result.
    pub fn scalar_poly_mul_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &[Target],
    ) -> Vec<ExtensionTarget<D>> {
        let mut result = vec![builder.constant_extension(F::ZERO.into()); a.len() + b.len() - 1];

        for i in 0..a.len() {
            for j in 0..b.len() {
                let aibj = builder.scalar_mul_ext(b[j], a[i]);
                result[i + j] = builder.add_extension(result[i + j], aibj);
            }
        }
        result
    }
}
