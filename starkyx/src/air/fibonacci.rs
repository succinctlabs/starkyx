use serde::{Deserialize, Serialize};

use super::parser::AirParser;
use super::{RAir, RAirData};
use crate::air::RoundDatum;
use crate::math::prelude::*;
use crate::trace::AirTrace;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FibonacciAir;

impl FibonacciAir {
    pub fn new() -> Self {
        Self {}
    }

    pub fn fibonacci<F: Field>(n: usize, x0: F, x1: F) -> F {
        (0..n).fold((x0, x1), |x, _| (x.1, x.0 + x.1)).1
    }

    pub fn generate_trace<F: Field>(x0: F, x1: F, num_rows: usize) -> AirTrace<F> {
        let trace_rows = (0..num_rows)
            .scan([x0, x1], |acc, _| {
                let tmp = *acc;
                acc[0] = tmp[1];
                acc[1] = tmp[0] + tmp[1];
                Some(tmp)
            })
            .flatten()
            .collect::<Vec<_>>();
        assert!(trace_rows.len() == num_rows * 2);
        AirTrace::from_rows(trace_rows, 2)
    }
}

impl Default for FibonacciAir {
    fn default() -> Self {
        Self::new()
    }
}

impl RAirData for FibonacciAir {
    fn constraint_degree(&self) -> usize {
        1
    }

    fn width(&self) -> usize {
        2
    }

    fn round_data(&self) -> Vec<RoundDatum> {
        vec![RoundDatum::new(self.width(), (0, 3), 0)]
    }

    fn num_public_inputs(&self) -> usize {
        3
    }
}

impl<AP: AirParser> RAir<AP> for FibonacciAir {
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
    fn test_fibonacci_air() {
        type F = GoldilocksField;

        let num_rows = 1 << 5usize;
        let air = FibonacciAir::new();

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(num_rows - 1, F::ZERO, F::ONE),
        ];
        let trace = FibonacciAir::generate_trace(F::ZERO, F::ONE, num_rows);

        for window in trace.windows() {
            assert_eq!(window.local_slice.len(), 2);
            let mut window_parser = TraceWindowParser::new(window, &[], &[], &public_inputs);
            assert_eq!(window_parser.local_slice().len(), 2);
            air.eval(&mut window_parser);
        }
    }
}
