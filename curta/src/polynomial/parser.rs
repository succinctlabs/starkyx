use itertools::Itertools;

use super::Polynomial;
use crate::air::parser::{AirParser, MulParser};

pub trait PolynomialParser: AirParser {
    fn zero_poly(&mut self) -> Polynomial<Self::Var> {
        Polynomial {
            coefficients: vec![self.zero()],
        }
    }

    fn constant_poly(&mut self, polynomial: &Polynomial<Self::Field>) -> Polynomial<Self::Var> {
        polynomial
            .coefficients
            .iter()
            .map(|x| self.constant(*x))
            .collect()
    }

    fn poly_add(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Var>,
    ) -> Polynomial<Self::Var> {
        a.coefficients
            .iter()
            .zip_longest(b.coefficients.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => self.add(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => *b,
            })
            .collect()
    }

    fn poly_sub(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Var>,
    ) -> Polynomial<Self::Var> {
        a.coefficients
            .iter()
            .zip_longest(b.coefficients.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => self.sub(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => self.neg(*b),
            })
            .collect()
    }

    fn poly_add_poly_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Field>,
    ) -> Polynomial<Self::Var> {
        a.coefficients
            .iter()
            .zip_longest(b.coefficients.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => self.add_const(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => self.constant(*b),
            })
            .collect()
    }

    fn poly_sub_poly_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Field>,
    ) -> Polynomial<Self::Var> {
        a.coefficients
            .iter()
            .zip_longest(b.coefficients.iter())
            .map(|x| match x {
                itertools::EitherOrBoth::Both(a, b) => self.sub_const(*a, *b),
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => self.constant(-*b),
            })
            .collect()
    }

    fn poly_scalar_add(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Var,
    ) -> Polynomial<Self::Var> {
        a.coefficients.iter().map(|x| self.add(*x, *b)).collect()
    }

    fn poly_scalar_sub(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Var,
    ) -> Polynomial<Self::Var> {
        a.coefficients.iter().map(|x| self.sub(*x, *b)).collect()
    }

    fn poly_sub_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Field,
    ) -> Polynomial<Self::Var> {
        a.coefficients
            .iter()
            .map(|x| self.sub_const(*x, *b))
            .collect()
    }

    fn poly_mul(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Var>,
    ) -> Polynomial<Self::Var> {
        let mut result = vec![self.zero(); a.coefficients.len() + b.coefficients.len() - 1];
        for (i, a) in a.coefficients.iter().enumerate() {
            for (j, b) in b.coefficients.iter().enumerate() {
                let ab = self.mul(*a, *b);
                result[i + j] = self.add(result[i + j], ab);
            }
        }
        Polynomial::from_coefficients(result)
    }

    fn poly_scalar_mul(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Var,
    ) -> Polynomial<Self::Var> {
        a.coefficients.iter().map(|x| self.mul(*x, *b)).collect()
    }

    fn poly_mul_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Field,
    ) -> Polynomial<Self::Var> {
        a.coefficients
            .iter()
            .map(|x| self.mul_const(*x, *b))
            .collect()
    }

    fn poly_mul_poly_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Field>,
    ) -> Polynomial<Self::Var> {
        let mut result = vec![self.zero(); a.coefficients.len() + b.coefficients.len() - 1];
        for (i, a) in a.coefficients.iter().enumerate() {
            for (j, b) in b.coefficients.iter().enumerate() {
                let ab = self.mul_const(*a, *b);
                result[i + j] = self.add(result[i + j], ab);
            }
        }
        Polynomial::from_coefficients(result)
    }
}

impl<'a, AP: PolynomialParser> PolynomialParser for MulParser<'a, AP> {
    fn zero_poly(&mut self) -> Polynomial<Self::Var> {
        self.parser.zero_poly()
    }

    fn constant_poly(&mut self, polynomial: &Polynomial<Self::Field>) -> Polynomial<Self::Var> {
        self.parser.constant_poly(polynomial)
    }

    fn poly_add(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Var>,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_add(a, b)
    }

    fn poly_sub(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Var>,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_sub(a, b)
    }

    fn poly_add_poly_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Field>,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_add_poly_const(a, b)
    }

    fn poly_sub_poly_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Field>,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_sub_poly_const(a, b)
    }

    fn poly_scalar_add(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Var,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_scalar_add(a, b)
    }

    fn poly_scalar_sub(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Var,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_scalar_sub(a, b)
    }

    fn poly_sub_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Field,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_sub_const(a, b)
    }

    fn poly_mul(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Var>,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_mul(a, b)
    }

    fn poly_scalar_mul(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Var,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_scalar_mul(a, b)
    }

    fn poly_mul_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Self::Field,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_mul_const(a, b)
    }

    fn poly_mul_poly_const(
        &mut self,
        a: &Polynomial<Self::Var>,
        b: &Polynomial<Self::Field>,
    ) -> Polynomial<Self::Var> {
        self.parser.poly_mul_poly_const(a, b)
    }
}
