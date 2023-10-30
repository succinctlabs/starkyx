use core::fmt::Debug;

use serde::de::DeserializeOwned;
use serde::Serialize;

use super::element::CubicElement;

/// Parameters for the cubic extension F[X]/(X^3 - X - 1)
pub trait CubicParameters<F>:
    'static + Sized + Copy + Clone + Send + Sync + PartialEq + Eq + Debug + Serialize + DeserializeOwned
{
    /// The Galois orbit of the generator.
    ///
    /// These are the roots of X^3 - X - 1 in the extension field not equal to X.
    const GALOIS_ORBIT: [CubicElement<F>; 2];
}
