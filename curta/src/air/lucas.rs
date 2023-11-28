use serde::{Deserialize, Serialize};

use super::parser::AirParser;
use super::{RAir, RAirData};
use crate::air::RoundDatum;
use crate::math::prelude::*;
use crate::trace::AirTrace;

/// Implements the lucas sequence `U(P, Q)` defined by the recurrence relation
/// `x_n = P * x_{n - 1} - Q * x_{n - 2}`.
///
/// This is a generalization of the Fibonacci sequence since `(P, Q) = (1, -1)` gives the Fibonacci
/// sequence. Just like the Fibonacci sequence, `x_0` and `x_1` must be defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LucasAir;

impl LucasAir {
    pub fn new() -> Self {
        Self {}
    }

    pub fn lucas<F: Field>(n: usize, x0: F, x1: F, p: F, q: F) -> F {
        (0..n).fold((x0, x1), |x, _| (x.1, p * x.1 - q * x.0)).1
    }

    /// Hello world.
    pub fn generate_trace<F: Field>(x0: F, x1: F, p: F, q: F, num_rows: usize) -> AirTrace<F> {
        let trace_rows = (0..num_rows)
            .scan([x0, x1, p, q], |acc, _| {
                let tmp = *acc;
                acc[0] = tmp[1]; // previous
                acc[1] = p * tmp[1] - q * tmp[0]; // current
                Some(tmp)
            })
            .flatten()
            .collect::<Vec<_>>();
        assert!(trace_rows.len() == num_rows * 4);
        AirTrace::from_rows(trace_rows, 4)
    }
}

impl Default for LucasAir {
    fn default() -> Self {
        Self::new()
    }
}

impl RAirData for LucasAir {
    fn constraint_degree(&self) -> usize {
        2
    }

    fn width(&self) -> usize {
        4
    }

    fn round_data(&self) -> Vec<RoundDatum> {
        vec![RoundDatum::new(self.width(), (0, 3), 0)]
    }

    fn num_public_inputs(&self) -> usize {
        5
    }
}

impl<AP: AirParser> RAir<AP> for LucasAir {
    fn eval(&self, parser: &mut AP) {
        // Check public inputs.
        let pis_constraints = [
            parser.sub(parser.local_slice()[0], parser.public_slice()[0]),
            parser.sub(parser.local_slice()[1], parser.public_slice()[1]),
            // parser.sub(parser.local_slice()[1], parser.global_slice()[2]),
        ];
        parser.constraint_first_row(pis_constraints[0]);
        parser.constraint_first_row(pis_constraints[1]);
        // parser.constraint_last_row(pis_constraints[2]);

        // x0' <- x1
        let first_col_constraint = parser.sub(parser.next_slice()[0], parser.local_slice()[1]);
        parser.constraint_transition(first_col_constraint);
        // x1' <- x0 + x1
        let second_col_constraint = {
            let tmp = parser.sub(parser.next_slice()[1], parser.local_slice()[0]);
            parser.sub(tmp, parser.local_slice()[1])
        };
        parser.constraint_transition(second_col_constraint);
    }

    fn eval_global(&self, _parser: &mut AP) {}
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::trace::window_parser::TraceWindowParser;

    #[test]
    fn test_lucas_air() {
        type F = GoldilocksField;

        let num_rows = 1 << 5usize;
        let air = LucasAir::new();

        let public_inputs = [
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ONE,
            LucasAir::lucas(num_rows - 1, F::ZERO, F::ONE, F::ONE, F::ONE),
        ];
        let trace = LucasAir::generate_trace(F::ZERO, F::ONE, num_rows, F::ONE, F::ONE);

        for window in trace.windows() {
            assert_eq!(window.local_slice.len(), 2);
            let mut window_parser = TraceWindowParser::new(window, &[], &[], &public_inputs);
            assert_eq!(window_parser.local_slice().len(), 2);
            air.eval(&mut window_parser);
        }
    }
}
