use itertools::Itertools;

use super::Polynomial;
use crate::curta::air::parser::AirParser;

#[derive(Debug)]
pub struct PolynomialParser<'a, P: AirParser> {
    pub parser: &'a mut P,
}

impl<'a, P: AirParser> PolynomialParser<'a, P> {
    pub fn new(parser: &'a mut P) -> Self {
        Self { parser }
    }

    pub fn constant(&mut self, polynomial: &Polynomial<P::Field>) -> Polynomial<P::Var> {
        polynomial
            .coefficients
            .iter()
            .map(|x| self.parser.constant(*x))
            .collect()
    }

    pub fn zero(&mut self) -> Polynomial<P::Var> {
        Polynomial::from_coefficients(vec![self.parser.zero()])
    }

    pub fn add(&mut self, a: &Polynomial<P::Var>, b: &Polynomial<P::Var>) -> Polynomial<P::Var> {
        a.coefficients
            .iter()
            .zip_longest(b.coefficients.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => self.parser.add(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => *b,
            })
            .collect()
    }

    pub fn sub(&mut self, a: &Polynomial<P::Var>, b: &Polynomial<P::Var>) -> Polynomial<P::Var> {
        a.coefficients
            .iter()
            .zip_longest(b.coefficients.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => self.parser.sub(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => self.parser.neg(*b),
            })
            .collect()
    }

    pub fn mul(&mut self, a: &Polynomial<P::Var>, b: &Polynomial<P::Var>) -> Polynomial<P::Var> {
        let mut result = vec![self.parser.zero(); a.coefficients.len() + b.coefficients.len() - 1];
        for (i, a) in a.coefficients.iter().enumerate() {
            for (j, b) in b.coefficients.iter().enumerate() {
                let ab = self.parser.mul(*a, *b);
                result[i + j] = self.parser.add(result[i + j], ab);
            }
        }
        Polynomial::from_coefficients(result)
    }

    pub fn scalar_sub(&mut self, a: &Polynomial<P::Var>, b: &P::Field) -> Polynomial<P::Var> {
        a.coefficients
            .iter()
            .map(|x| self.parser.scalar_sub(*x, *b))
            .collect()
    }

    pub fn scalar_mul(&mut self, a: &Polynomial<P::Var>, b: &P::Field) -> Polynomial<P::Var> {
        a.coefficients
            .iter()
            .map(|x| self.parser.scalar_mul(*x, *b))
            .collect()
    }

    pub fn scalar_poly_add(
        &mut self,
        a: &Polynomial<P::Var>,
        b: &Polynomial<P::Field>,
    ) -> Polynomial<P::Var> {
        a.coefficients
            .iter()
            .zip_longest(b.coefficients.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => self.parser.scalar_add(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => self.parser.constant(*b),
            })
            .collect()
    }

    pub fn scalar_poly_sub(
        &mut self,
        a: &Polynomial<P::Var>,
        b: &Polynomial<P::Field>,
    ) -> Polynomial<P::Var> {
        a.coefficients
            .iter()
            .zip_longest(b.coefficients.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => self.parser.scalar_sub(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => self.parser.constant(-*b),
            })
            .collect()
    }

    pub fn scalar_poly_mul(
        &mut self,
        a: &Polynomial<P::Var>,
        b: &Polynomial<P::Field>,
    ) -> Polynomial<P::Var> {
        let mut result = vec![self.parser.zero(); a.coefficients.len() + b.coefficients.len() - 1];
        for (i, a) in a.coefficients.iter().enumerate() {
            for (j, b) in b.coefficients.iter().enumerate() {
                let ab = self.parser.scalar_mul(*a, *b);
                result[i + j] = self.parser.add(result[i + j], ab);
            }
        }
        Polynomial::from_coefficients(result)
    }
}
