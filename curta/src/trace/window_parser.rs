use super::window::TraceWindow;
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::math::prelude::*;
use crate::polynomial::parser::PolynomialParser;

#[derive(Debug, Clone)]
pub struct TraceWindowParser<'a, T> {
    window: TraceWindow<'a, T>,
    challenge_slice: &'a [T],
    global_slice: &'a [T],
    public_slice: &'a [T],
}

impl<'a, T> TraceWindowParser<'a, T> {
    pub fn new(
        window: TraceWindow<'a, T>,
        challenge_slice: &'a [T],
        global_slice: &'a [T],
        public_slice: &'a [T],
    ) -> Self {
        Self {
            window,
            challenge_slice,
            global_slice,
            public_slice,
        }
    }
}

impl<'a, F: Field> AirParser for TraceWindowParser<'a, F> {
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

    fn global_slice(&self) -> &[Self::Var] {
        self.global_slice
    }

    fn public_slice(&self) -> &[Self::Var] {
        self.public_slice
    }

    fn constraint(&mut self, constraint: Self::Var) {
        assert_eq!(
            constraint,
            F::ZERO,
            "Nonzero constraint: {:?} at row: {}",
            constraint,
            self.window.row
        );
    }

    fn constraint_transition(&mut self, constraint: Self::Var) {
        if !self.window.is_last_row {
            assert_eq!(
                constraint,
                F::ZERO,
                "Nonzero constraint: {:?} at row: {}",
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
                "Nonzero constraint at first row: {constraint:?}"
            );
        }
    }

    fn constraint_last_row(&mut self, constraint: Self::Var) {
        if self.window.is_last_row {
            assert_eq!(
                constraint,
                F::ZERO,
                "Nonzero constraint at last row: {constraint:?}"
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

impl<'a, F: Field> PolynomialParser for TraceWindowParser<'a, F> {}

impl<'a, F: Field, E: CubicParameters<F>> CubicParser<E> for TraceWindowParser<'a, F> {}
