use plonky2::field::types::Field; 


/// A wrapper around a vector of field elements that implements polynomial operations.
#[derive(Debug, Clone)]
pub struct Polynomial<F>{
    coefficients: Vec<F>,
}

impl<F : Field> Polynomial<F> {
    fn new_from_vec(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }
    fn new_from_slice(coefficients: &[F]) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
        }
    }
}


#[derive(Debug, Clone, Copy)]
pub struct PolynomialOperations<F>{
    _marker: core::marker::PhantomData<F>,
}

impl<F : Field> PolynomialOperations<F> {

    /// Polynomial addition
    fn add(a : &[F], b : &[F]) -> Vec<F> {
        assert!(a.len() == b.len());
        a.iter().zip(b.iter()).map(|(a, b)| *a + *b).collect()
    }

    /// Computes polynomial addition and assigns to a
    fn add_assign(a: &mut [F], b: &[F]) {
        assert!(a.len() == b.len());
        for (a, b) in a.iter_mut().zip(b.iter()) {
            *a += *b;
        }
    }

    /// Multiply a polynomial by a scalar
    fn mul_scalar(a : &[F], b : F) -> Vec<F> {
        a.iter().map(|a| *a * b).collect()
    }

    /// Multiply a polynomial by a scalar and assign to a
    fn mul_scalar_assign(a: &mut [F], b: F) {
        for a in a.iter_mut() {
            *a *= b;
        }
    }

    // Polynomial multiplication
    fn mul(a : &[F], b : &[F]) -> Vec<F> {
        assert_eq!(a.len(), b.len());

        let mut result = vec![F::ZERO; a.len() + b.len() - 1];

        for (i, a) in a.iter().enumerate() {
            for (j, b) in b.iter().enumerate() {
                result[i + j] += *a * *b;
            }
        }
        result
    }

    // Computes the polynomial multiplication and assigns result to a
    fn mu_assign(a : &mut [F], b : &[F]) {
        assert_eq!(a.len(), b.len());

        let result = Self::add(a, b);
        for (i, res) in result.iter().enumerate() {
            a[i] = *res;
        }
    }

    fn root_witness(a : &[F], root : F) -> Vec<F> {
        vec![]
    }
        
        
}