use core::fmt::{Debug, Display};

use crate::math::prelude::*;
use crate::trace::TraceWindow;

pub trait AirParser: Send + Sized {
    type Field: Field;

    type Var: Debug + Copy + 'static;

    fn local_slice(&self) -> &[Self::Var];
    fn next_slice(&self) -> &[Self::Var];
    fn challenge_slice(&self) -> &[Self::Var];
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

    fn assert_eq(&mut self, a: Self::Var, b: Self::Var) {
        let c = self.sub(a, b);
        self.constraint(c);
    }

    fn assert_eq_transition(&mut self, a: Self::Var, b: Self::Var) {
        let c = self.sub(a, b);
        self.constraint_transition(c);
    }
}

#[derive(Debug, Clone)]
pub struct TraceWindowParser<'a, F: Field> {
    window: TraceWindow<'a, F>,
    challenge_slice: &'a [F],
    public_slice: &'a [F],
}

impl<'a, F: Field> TraceWindowParser<'a, F> {
    pub fn new(
        window: TraceWindow<'a, F>,
        challenge_slice: &'a [F],
        public_slice: &'a [F],
    ) -> Self {
        Self {
            window,
            challenge_slice,
            public_slice,
        }
    }
}

impl<'a, F: Field + Display> AirParser for TraceWindowParser<'a, F> {
    type Field = F;

    type Var = F;

    fn local_slice(&self) -> &[Self::Var] {
        self.window.local_slice
    }

    fn next_slice(&self) -> &[Self::Var] {
        self.window.next_slice
    }

    fn challenge_slice(&self) -> &[Self::Var] {
        self.challenge_slice
    }

    fn public_slice(&self) -> &[Self::Var] {
        self.public_slice
    }

    fn constraint(&mut self, constraint: Self::Var) {
        assert_eq!(
            constraint,
            F::ZERO,
            "Nonzero constraint: {} at row: {}",
            constraint,
            self.window.row
        );
    }

    fn constraint_transition(&mut self, constraint: Self::Var) {
        if !self.window.is_last_row {
            assert_eq!(
                constraint,
                F::ZERO,
                "Nonzero constraint: {} at row: {}",
                constraint,
                self.window.row
            );
        }
    }

    fn constraint_first_row(&mut self, constraint: Self::Var) {
        if self.window.is_first_row {
            assert_eq!(
                constraint,
                F::ZERO,
                "Nonzero constraint at first row: {}",
                constraint
            );
        }
    }

    fn constraint_last_row(&mut self, constraint: Self::Var) {
        if self.window.is_last_row {
            assert_eq!(
                constraint,
                F::ZERO,
                "Nonzero constraint at last row: {}",
                constraint
            );
        }
    }

    fn constant(&mut self, value: Self::Field) -> Self::Var {
        value
    }

    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        a + b
    }

    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        a - b
    }

    fn neg(&mut self, a: Self::Var) -> Self::Var {
        -a
    }

    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        a * b
    }
}
