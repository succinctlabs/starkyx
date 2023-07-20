use super::element::CubicElement;

/// Parameters for the cubic extension F[X]/(X^3 - X - 1)
pub trait CubicParameters<F>:
    'static + Sized + Copy + Clone + Send + Sync + PartialEq + Eq + std::fmt::Debug
{
    /// The Galois orbit of the generator.
    ///
    /// These are the roots of X^3 - X - 1 in the extension field not equal to X.
    const GALOIS_ORBIT: [CubicElement<F>; 2];
}
