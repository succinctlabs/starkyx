//! Implementations of polynomial operations
//!
//! This module includes:
//!
//! - PolynomialOps: a struct that implements polynomial operations on slices
//!   in a generic way. This is useful for general purpose operations when we don't
//!  need to use a wrapper
//! - Polynomial: a wrapper around a vector of field elements that implements polynomial
//!     arithmetic with the usual Add, Mul, Sub operators.
//! - PolynomialGadget: a struct that implements polynomial operations on a circuit.
//!    the operations take a mutable reference to a circuit builder and the
//!    respective targets, and performs the polynomial operations in-circuit.
//!                    
//!

use core::iter;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

/// A wrapper around a vector of field elements that implements polynomial operations.
///
/// Used to represent polynomials when computing with traces
///
#[derive(Debug, Clone)]
pub struct Polynomial<F> {
    coefficients: Vec<F>,
}

/// General purpose polynomial operations on slices
///
/// Usefeul for general purpose operations when we don't need to use a wrapper
///
/// All operations here assume input polynomials are of the same degree
#[derive(Debug, Clone, Copy)]
pub struct PolynomialOps;

#[derive(Debug, Clone, Copy)]
pub struct PolynomialGadget;

impl<T: Clone> Polynomial<T> {
    pub fn new_from_vec(coefficients: Vec<T>) -> Self {
        Self { coefficients }
    }

    pub fn new_from_slice(coefficients: &[T]) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
        }
    }

    pub fn into_vec(self) -> Vec<T> {
        self.coefficients
    }

    pub fn coefficients(&self) -> Vec<T> {
        self.coefficients.clone()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.coefficients
    }

    pub fn as_mut_vec(&mut self) -> &mut Vec<T> {
        &mut self.coefficients
    }

    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    pub fn constant(value: T) -> Self {
        Self {
            coefficients: vec![value],
        }
    }
}

impl<T: Default + Clone> Polynomial<T> {
    pub fn constant_monomial(c: T, degree: usize) -> Self {
        let mut coefficients = vec![T::default(); degree + 1];
        coefficients[degree] = c;
        Self { coefficients }
    }
}

impl PolynomialOps {
    /// Polynomial addition
    pub fn add<T>(a: &[T], b: &[T]) -> Vec<T>
    where
        T: Add<Output = T> + Copy + Default,
    {
        a.iter()
            .zip_longest(b.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => *a + *b,
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => *b,
            })
            .collect()
    }

    /// Polynomial addition with assignment
    ///
    /// Assumes a.len() >= b.len()
    pub fn add_assign<T>(a: &mut [T], b: &[T])
    where
        T: AddAssign + Copy,
    {
        debug_assert!(a.len() >= b.len(), "Expects a.len() >= b.len()");
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += *b);
    }

    pub fn neg<T>(a: &[T]) -> Vec<T>
    where
        T: Neg<Output = T> + Copy,
    {
        a.iter().map(|x| -*x).collect()
    }

    /// Polynomial subtraction
    pub fn sub<T>(a: &[T], b: &[T]) -> Vec<T>
    where
        T: Sub<Output = T> + Copy + Neg<Output = T>,
    {
        a.iter()
            .zip_longest(b.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => *a - *b,
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => -*b,
            })
            .collect()
    }

    /// Polynomial subtraction with assignment
    ///
    /// Assumes a.len() >= b.len()
    pub fn sub_assign<T>(a: &mut [T], b: &[T])
    where
        T: SubAssign + Copy,
    {
        debug_assert!(a.len() >= b.len());
        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a -= *b);
    }

    /// Polynomial multiplication
    pub fn mul<T>(a: &[T], b: &[T]) -> Vec<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy + Default + Add<Output = T>,
    {
        let mut result = vec![T::default(); a.len() + b.len() - 1];
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[i + j] = result[i + j] + a[i] * b[j];
            }
        }
        result
    }

    pub fn scalar_poly_add<T, S>(a: &[T], b: &[S]) -> Vec<T>
    where
        T: Add<S, Output = T> + Copy + Default,
        S: Copy,
    {
        debug_assert!(a.len() == b.len());
        a.iter().zip(b.iter()).map(|(a, b)| *a + *b).collect()
    }

    pub fn scalar_sub<T, S>(a: &[T], b: &S) -> Vec<T>
    where
        T: Sub<S, Output = T> + Copy,
        S: Copy,
    {
        a.iter().map(|x| *x - *b).collect()
    }

    pub fn scalar_poly_sub<T, S>(a: &[T], b: &[S]) -> Vec<T>
    where
        T: Sub<S, Output = T> + Copy + Default,
        S: Copy,
    {
        debug_assert!(a.len() == b.len());
        a.iter().zip(b.iter()).map(|(a, b)| *a - *b).collect()
    }

    /// Multiply a polynomial by a scalar
    pub fn scalar_mul<T, S>(a: &[T], b: &S) -> Vec<T>
    where
        T: Mul<S, Output = T> + Copy,
        S: Copy,
    {
        a.iter().map(|x| *x * *b).collect()
    }

    /// Multiply a polynomial by a polynomial with a scalar coefficients
    pub fn scalar_poly_mul<T, S>(a: &[T], b: &[S]) -> Vec<T>
    where
        T: Mul<S, Output = T> + Copy + Default + AddAssign,
        S: Copy,
    {
        let mut result = vec![T::default(); a.len() + b.len() - 1];
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[i + j] += a[i] * b[j];
            }
        }
        result
    }

    /// Evaluate the polynomial at a point
    pub fn eval<T, F>(a: &[T], x: &F) -> T
    where
        T: Add<Output = T> + Mul<F, Output = T> + Copy + iter::Sum,
        F: Field,
    {
        let powers = x.powers();

        a.iter().zip(powers).map(|(a, x)| *a * x).sum()
    }

    /// Extract the quotient s(x) of a(x) such that
    /// (x-r)s(x) when r is a root of a(x)
    pub fn root_quotient<F: Field>(a: &[F], r: &F) -> Vec<F> {
        assert_eq!(PolynomialOps::eval(a, r), F::ZERO);
        let mut result = Vec::with_capacity(a.len() - 2);
        let r_inverse = if *r == F::ZERO { F::ZERO } else { r.inverse() };

        result.push(-a[0] * r_inverse);
        for i in 1..a.len() - 1 {
            let element = result[i - 1] - a[i];
            result.push(element * r_inverse);
        }
        result
    }
}

//
// Arithmetic operations for the struct Polynomial<T>
//

impl<T> Polynomial<T> {
    pub fn eval<F>(&self, x: F) -> T
    where
        T: Add<Output = T> + Mul<F, Output = T> + Copy + iter::Sum + Default,
        F: Field,
    {
        PolynomialOps::eval(self.as_slice(), &x)
    }
}

impl<F: Field> Polynomial<F> {
    pub fn root_quotient(&self, r: F) -> Self {
        Self::new_from_vec(PolynomialOps::root_quotient(self.as_slice(), &r))
    }

    pub fn x() -> Self {
        Self::new_from_vec(vec![F::ZERO, F::ONE])
    }

    pub fn x_n(n: usize) -> Self {
        let mut result = vec![F::ZERO; n + 1];
        result[n] = F::ONE;
        Self::new_from_vec(result)
    }
}

impl<T: Add<Output = T> + Copy + Default> Add for Polynomial<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new_from_vec(PolynomialOps::add(self.as_slice(), other.as_slice()))
    }
}
impl<T: Add<Output = T> + Copy + Default> Add for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn add(self, other: Self) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::add(self.as_slice(), other.as_slice()))
    }
}

impl<T: Neg<Output = T> + Copy> Neg for Polynomial<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new_from_vec(PolynomialOps::neg(self.as_slice()))
    }
}

impl<T: Sub<Output = T> + Neg<Output = T> + Copy + Default> Sub for Polynomial<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new_from_vec(PolynomialOps::sub(self.as_slice(), other.as_slice()))
    }
}

impl<T: Sub<Output = T> + Neg<Output = T> + Copy + Default> Sub for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn sub(self, other: Self) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::sub(self.as_slice(), other.as_slice()))
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + Copy + Default> Mul for Polynomial<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new_from_vec(PolynomialOps::mul(self.as_slice(), other.as_slice()))
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + Copy + Default> Mul for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn mul(self, other: Self) -> Polynomial<T> {
        Polynomial::new_from_vec(PolynomialOps::mul(self.as_slice(), other.as_slice()))
    }
}

impl<T: Default + Clone> Default for Polynomial<T> {
    fn default() -> Self {
        Self::new_from_vec(vec![T::default()])
    }
}

// Polynomial gadget for the circuit builder with Target as inputs
impl PolynomialGadget {
    pub fn constant<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[F],
    ) -> Vec<Target> {
        a.iter().map(|x| builder.constant(*x)).collect()
    }

    pub fn add<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[Target],
        b: &[Target],
    ) -> Vec<Target> {
        debug_assert!(a.len() == b.len());
        let len = a.len();

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(builder.add(a[i], b[i]));
        }
        result
    }

    pub fn neg<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[Target],
    ) -> Vec<Target> {
        a.iter().map(|x| builder.neg(*x)).collect()
    }

    /// Polynomial subtraction
    pub fn sub<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[Target],
        b: &[Target],
    ) -> Vec<Target> {
        debug_assert!(a.len() == b.len());
        let len = a.len();

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(builder.sub(a[i], b[i]));
        }
        result
    }

    /// Polynomial multiplication
    pub fn mul<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[Target],
        b: &[Target],
    ) -> Vec<Target> {
        let mut result = vec![builder.constant(F::ZERO); a.len() + b.len() - 1];

        for i in 0..a.len() {
            for j in 0..b.len() {
                let aibj = builder.mul(a[i], b[j]);
                result[i + j] = builder.add(result[i + j], aibj);
            }
        }

        result
    }

    pub fn scalar_mul<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[Target],
        b: &Target,
    ) -> Vec<Target> {
        a.iter().map(|x| builder.mul(*b, *x)).collect()
    }
}

// Polynomial operations for circuit builder with ExtensionTarget as inputs

impl PolynomialGadget {
    pub fn constant_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[F::Extension],
    ) -> Vec<ExtensionTarget<D>> {
        a.iter().map(|x| builder.constant_extension(*x)).collect()
    }

    pub fn add_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &[ExtensionTarget<D>],
    ) -> Vec<ExtensionTarget<D>> {
        debug_assert!(a.len() == b.len());
        let len = a.len();

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(builder.add_extension(a[i], b[i]));
        }
        result
    }

    /// Polynomial subtraction
    pub fn sub_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &[ExtensionTarget<D>],
    ) -> Vec<ExtensionTarget<D>> {
        debug_assert!(a.len() == b.len());
        let len = a.len();

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(builder.sub_extension(a[i], b[i]));
        }
        result
    }

    /// Polynomial multiplication
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

    pub fn scalar_mul_extension<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &Target,
    ) -> Vec<ExtensionTarget<D>> {
        a.iter().map(|x| builder.scalar_mul_ext(*b, *x)).collect()
    }

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

#[cfg(test)]
mod tests {
    use num::BigUint;
    use plonky2::field::extension::quadratic::QuadraticExtension;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, Sample};
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::*;

    #[test]
    fn test_default_value() {
        assert_eq!(BigUint::default(), BigUint::from(0u32));
        assert_eq!(GoldilocksField::default(), GoldilocksField::ZERO);

        let poly_zero = Polynomial::<GoldilocksField>::default();
        let x = GoldilocksField::rand();
        assert_eq!(poly_zero.eval(x), GoldilocksField::ZERO);
    }

    fn get_random_poly<F: Sample + Default + Clone>(degree: usize) -> Polynomial<F> {
        Polynomial::new_from_vec((0..=degree).map(|_| F::rand()).collect::<Vec<_>>())
    }

    #[test]
    fn test_evaluation() {
        let a = get_random_poly::<GoldilocksField>(10);
        let x = GoldilocksField::rand();

        let a_coeff = a.coefficients();
        let powers = x.powers().take(11);

        assert_eq!(a_coeff.len(), 11);

        let eval = a_coeff
            .iter()
            .zip(powers)
            .map(|(a, x)| *a * x)
            .sum::<GoldilocksField>();

        assert_eq!(eval, a.eval(x));
    }

    #[test]
    fn test_poly_add() {
        // Test with field elements
        // Use propery based testing with evaluation
        let deg_a = 15;
        let deg_b = 15;
        let num_tests = 100;

        for _ in 0..num_tests {
            let a = get_random_poly::<GoldilocksField>(deg_a);
            let b = get_random_poly::<GoldilocksField>(deg_b);

            let poly_sum = &a + &b;

            let x_eval = GoldilocksField::rand();
            assert_eq!(a.eval(x_eval) + b.eval(x_eval), poly_sum.eval(x_eval));
        }
    }

    #[test]
    fn test_poly_neg() {
        let deg_a = 15;
        let num_tests = 100;

        for _ in 0..num_tests {
            let a = get_random_poly::<GoldilocksField>(deg_a);
            let neg_a = -a.clone();
            let neg_plus_a = &neg_a + &a;

            let x = GoldilocksField::rand();
            assert_eq!(neg_plus_a.eval(x), GoldilocksField::ZERO);
            assert_eq!(neg_a.eval(x), -a.eval(x));
        }
    }

    #[test]
    fn test_poly_sub() {
        // Test with field elements
        // Use propery based testing with evaluation
        let deg_a = 1;
        let deg_b = 4;
        let num_tests = 100;

        for _ in 0..num_tests {
            let a = get_random_poly::<GoldilocksField>(deg_a);
            let b = get_random_poly::<GoldilocksField>(deg_b);

            let poly_diff = &a - &b;

            let x = GoldilocksField::rand();
            assert_eq!(a.eval(x) - b.eval(x), poly_diff.eval(x));
        }
    }

    #[test]
    fn test_poly_mul() {
        // Test with field elements
        // Use propery based testing with evaluation
        let deg_a = 10;
        let deg_b = 14;
        let num_tests = 100;

        let zero = Polynomial::default();
        for _ in 0..num_tests {
            let a = get_random_poly::<GoldilocksField>(deg_a);
            let b = get_random_poly::<GoldilocksField>(deg_b);

            let poly_mul = &a * &b;
            let x = GoldilocksField::rand();
            assert_eq!(a.eval(x) * b.eval(x), poly_mul.eval(x));

            let poly_mul_zero = &a * &zero;
            assert_eq!(poly_mul_zero.eval(x), GoldilocksField::ZERO);
        }
    }

    #[test]
    fn test_extract_root_quotient() {
        let n_degree = 34;
        let num_tests = 10;

        for _ in 0..num_tests {
            let a = get_random_poly::<GoldilocksField>(n_degree);

            let x = GoldilocksField::rand();
            let val = Polynomial::constant(a.eval(x));

            let p = a - val;
            let q = p.root_quotient(x);

            let y = GoldilocksField::rand();
            assert_eq!((y - x) * q.eval(y), p.eval(y));
        }
    }

    #[test]
    fn test_scalar_operations() {
        let num_tests = 10;
        type F2 = QuadraticExtension<GoldilocksField>;

        for _ in 0..num_tests {
            let a = get_random_poly::<F2>(15);
            let b = get_random_poly::<F2>(13);
            let _s = get_random_poly::<GoldilocksField>(14);

            let x = F2::rand();
            assert_eq!((&a + &b).eval(x), a.eval(x) + b.eval(x));
            assert_eq!((&a * &b).eval(x), a.eval(x) * b.eval(x));
        }
    }

    fn assert_slice_ext_equal<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[ExtensionTarget<D>],
        b: &[ExtensionTarget<D>],
    ) {
        //assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let diff = builder.sub_extension(a[i], b[i]);
            let entries = diff.to_target_array();
            for e in entries {
                builder.assert_zero(e);
            }
        }
    }

    fn assert_slice_equal<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        a: &[Target],
        b: &[Target],
    ) {
        //assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let diff = builder.sub(a[i], b[i]);
            builder.assert_zero(diff);
        }
    }

    fn set_targets<F: Field>(pw: &mut PartialWitness<F>, targets: &[Target], values: &[F]) {
        assert_eq!(targets.len(), values.len());
        for i in 0..targets.len() {
            pw.set_target(targets[i], values[i]);
        }
    }

    #[test]
    fn test_builder_operations() {
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        const D: usize = 2;

        let d = 15;

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config_rec);

        let a = get_random_poly::<F>(d);
        let b = get_random_poly::<F>(d);
        let expected_sum = &a + &b;
        let expect_diff = &a - &b;
        let expected_mul = &a * &b;
        let expected_zeros = Polynomial::new_from_vec(vec![F::ZERO; 2 * d + 1]);

        let x = F::rand();
        assert_eq!(expected_sum.eval(x), a.eval(x) + b.eval(x));
        assert_eq!(expect_diff.eval(x), a.eval(x) - b.eval(x));
        assert_eq!(expected_mul.eval(x), a.eval(x) * b.eval(x));

        let input_1 = builder.add_virtual_targets(d + 1);
        let input_2 = builder.add_virtual_targets(d + 1);
        let output_sum = builder.add_virtual_targets(d + 1);
        let output_diff = builder.add_virtual_targets(d + 1);
        let output_mul = builder.add_virtual_targets(2 * d + 1);
        let zeros = builder.add_virtual_targets(2 * d + 1);

        let sum = PolynomialGadget::add(&mut builder, &input_1, &input_2);
        assert_slice_equal(&mut builder, &sum, &output_sum);

        let diff = PolynomialGadget::sub(&mut builder, &input_1, &input_2);
        assert_slice_equal(&mut builder, &diff, &output_diff);

        let mul = PolynomialGadget::mul(&mut builder, &input_1, &input_2);
        assert_slice_equal(&mut builder, &mul, &output_mul);

        let mut pw = PartialWitness::<F>::new();

        set_targets(&mut pw, &input_1, a.as_slice());
        set_targets(&mut pw, &input_2, b.as_slice());
        set_targets(&mut pw, &output_sum, expected_sum.as_slice());
        set_targets(&mut pw, &output_diff, expect_diff.as_slice());
        set_targets(&mut pw, &output_mul, expected_mul.as_slice());
        set_targets(&mut pw, &zeros, expected_zeros.as_slice());

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();

        data.verify(proof).unwrap();
    }

    #[test]
    fn test_builder_extension_operations() {
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FE = <C as GenericConfig<D>>::FE;
        const D: usize = 2;

        let d = 15;

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config_rec);

        let a = get_random_poly::<FE>(d);
        let b = get_random_poly::<FE>(d);
        let expected_sum = &a + &b;
        let expect_diff = &a - &b;
        let expected_mul = &a * &b;
        let expected_zeros = Polynomial::new_from_vec(vec![FE::ZERO; 2 * d + 1]);

        let x = FE::rand();
        assert_eq!(expected_sum.eval(x), a.eval(x) + b.eval(x));
        assert_eq!(expect_diff.eval(x), a.eval(x) - b.eval(x));
        assert_eq!(expected_mul.eval(x), a.eval(x) * b.eval(x));

        let input_1 = builder.add_virtual_extension_targets(d + 1);
        let input_2 = builder.add_virtual_extension_targets(d + 1);
        let output_sum = builder.add_virtual_extension_targets(d + 1);
        let output_diff = builder.add_virtual_extension_targets(d + 1);
        let output_mul = builder.add_virtual_extension_targets(2 * d + 1);
        let zeros = builder.add_virtual_extension_targets(2 * d + 1);

        let sum = PolynomialGadget::add_extension(&mut builder, &input_1, &input_2);
        assert_slice_ext_equal(&mut builder, &sum, &output_sum);

        let diff = PolynomialGadget::sub_extension(&mut builder, &input_1, &input_2);
        assert_slice_ext_equal(&mut builder, &diff, &output_diff);

        let mul = PolynomialGadget::mul_extension(&mut builder, &input_1, &input_2);
        assert_slice_ext_equal(&mut builder, &mul, &output_mul);

        let mut pw = PartialWitness::<F>::new();

        pw.set_extension_targets(&input_1, a.as_slice());
        pw.set_extension_targets(&input_2, b.as_slice());
        pw.set_extension_targets(&output_sum, expected_sum.as_slice());
        pw.set_extension_targets(&output_diff, expect_diff.as_slice());
        pw.set_extension_targets(&output_mul, expected_mul.as_slice());
        pw.set_extension_targets(&zeros, expected_zeros.as_slice());

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();

        data.verify(proof).unwrap();
    }
}
