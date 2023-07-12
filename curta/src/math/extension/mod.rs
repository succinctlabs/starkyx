use core::slice;

use super::algebra::Algebra;
use super::field::Field;

pub mod cubic;

pub use cubic::parameters::CubicParameters;
/// A ring extension of a field with a fixed basis
pub trait Extension<F: Field>: Algebra<F> {
    /// The dimension (i.e. degree) of the extension
    const D: usize;

    /// An element from a collection of coefficients in the field
    fn from_base_slice(elements: &[F]) -> Self;

    /// The coefficients of the element with respect to the fixed basis
    fn as_base_slice(&self) -> &[F];
}

/// A ring extension of a field with a fixed basis
pub trait ExtensionField<F: Field>: Extension<F> + Field {}

impl<F: Field> Extension<F> for F {
    const D: usize = 1;

    fn from_base_slice(elements: &[F]) -> Self {
        elements[0]
    }

    fn as_base_slice(&self) -> &[F] {
        slice::from_ref(self)
    }
}

impl<F: Field> ExtensionField<F> for F {}
