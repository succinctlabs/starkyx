pub mod parser;

/// A wrapper around a vector of elements to represent a polynomial.
#[derive(Debug, Clone)]
pub struct Polynomial<T> {
    pub coefficients: Vec<T>,
}

impl<T: Clone> Polynomial<T> {
    pub fn from_coefficients(coefficients: Vec<T>) -> Self {
        Self { coefficients }
    }

    pub fn from_slice(coefficients: &[T]) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
        }
    }
}

impl<T> FromIterator<T> for Polynomial<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            coefficients: iter.into_iter().collect(),
        }
    }
}
