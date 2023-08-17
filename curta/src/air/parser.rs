use core::fmt::Debug;

use super::extension::cubic::CubicParser;
use crate::math::prelude::*;

pub trait AirParser: Sized {
    type Field: Field;

    type Var: Debug + Copy + 'static;

    fn local_slice(&self) -> &[Self::Var];
    fn next_slice(&self) -> &[Self::Var];
    fn challenge_slice(&self) -> &[Self::Var];
    fn global_slice(&self) -> &[Self::Var];
    fn public_slice(&self) -> &[Self::Var];

    fn constraint(&mut self, constraint: Self::Var);
    fn constraint_transition(&mut self, constraint: Self::Var);
    fn constraint_first_row(&mut self, constraint: Self::Var);
    fn constraint_last_row(&mut self, constraint: Self::Var);

    fn constant(&mut self, value: Self::Field) -> Self::Var;

    /// Add two vars while potantially updating the internal state
    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    /// Subtract two vars while potantially updating the internal state
    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    /// Negate a var while potantially updating the internal state
    fn neg(&mut self, a: Self::Var) -> Self::Var;

    /// Multiply two vars while potantially updating the internal state
    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    fn add_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.add(a, b)
    }

    fn sub_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.sub(a, b)
    }

    fn mul_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.mul(a, b)
    }

    fn one(&mut self) -> Self::Var {
        self.constant(Self::Field::ONE)
    }

    fn zero(&mut self) -> Self::Var {
        self.constant(Self::Field::ZERO)
    }

    fn sum(&mut self, elements: &[Self::Var]) -> Self::Var {
        elements
            .iter()
            .fold(self.zero(), |acc, x| self.add(acc, *x))
    }

    fn assert_eq(&mut self, a: Self::Var, b: Self::Var) {
        let c = self.sub(a, b);
        self.constraint(c);
    }

    fn assert_eq_transition(&mut self, a: Self::Var, b: Self::Var) {
        let c = self.sub(a, b);
        self.constraint_transition(c);
    }
}

#[derive(Debug)]
pub struct MulParser<'a, AP: AirParser> {
    pub parser: &'a mut AP,
    pub multiplier: AP::Var,
}

impl<'a, AP: AirParser> MulParser<'a, AP> {
    pub fn new(parser: &'a mut AP, multiplier: AP::Var) -> Self {
        Self { parser, multiplier }
    }
}

impl<'a, AP: AirParser> AirParser for MulParser<'a, AP> {
    type Field = AP::Field;
    type Var = AP::Var;

    fn local_slice(&self) -> &[Self::Var] {
        self.parser.local_slice()
    }

    fn next_slice(&self) -> &[Self::Var] {
        self.parser.next_slice()
    }

    fn challenge_slice(&self) -> &[Self::Var] {
        self.parser.challenge_slice()
    }

    fn global_slice(&self) -> &[Self::Var] {
        self.parser.global_slice()
    }

    fn public_slice(&self) -> &[Self::Var] {
        self.parser.public_slice()
    }

    fn constraint(&mut self, constraint: Self::Var) {
        let constr = self.parser.mul(constraint, self.multiplier);
        self.parser.constraint(constr);
    }

    fn constraint_transition(&mut self, constraint: Self::Var) {
        let constr = self.parser.mul(constraint, self.multiplier);
        self.parser.constraint_transition(constr);
    }

    fn constraint_first_row(&mut self, constraint: Self::Var) {
        let constr = self.parser.mul(constraint, self.multiplier);
        self.parser.constraint_first_row(constr);
    }

    fn constraint_last_row(&mut self, constraint: Self::Var) {
        let constr = self.parser.mul(constraint, self.multiplier);
        self.parser.constraint_last_row(constr);
    }

    fn constant(&mut self, value: Self::Field) -> Self::Var {
        self.parser.constant(value)
    }

    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.parser.add(a, b)
    }

    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.parser.sub(a, b)
    }

    fn neg(&mut self, a: Self::Var) -> Self::Var {
        self.parser.neg(a)
    }

    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.parser.mul(a, b)
    }

    fn add_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        self.parser.add_const(a, b)
    }

    fn sub_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        self.parser.sub_const(a, b)
    }

    fn mul_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        self.parser.mul_const(a, b)
    }
}

// TODO: implement parser specific functions
impl<'a, AP: CubicParser<E>, E: CubicParameters<AP::Field>> CubicParser<E> for MulParser<'a, AP> {}
