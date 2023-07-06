use super::parser::AirParser;
use super::RAir;
use crate::math::prelude::*;
use crate::trace::StarkTrace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FibonacciAir;

impl FibonacciAir {
    pub fn new() -> Self {
        Self {}
    }

    pub fn fibonacci<F: Field>(n: usize, x0: F, x1: F) -> F {
        (0..n).fold((x0, x1), |x, _| (x.1, x.0 + x.1)).1
    }

    pub fn generate_trace<F: Field>(x0: F, x1: F, num_rows: usize) -> StarkTrace<F> {
        let trace_rows = (0..num_rows)
            .scan([x0, x1, F::ZERO, F::ONE], |acc, _| {
                let tmp = *acc;
                acc[0] = tmp[1];
                acc[1] = tmp[0] + tmp[1];
                acc[2] = tmp[2] + F::ONE;
                acc[3] = tmp[3] + F::ONE;
                Some(tmp)
            })
            .flat_map(|arr| arr)
            .collect::<Vec<_>>();
        assert!(trace_rows.len() == num_rows * 4);
        let mut trace = StarkTrace::from_rows(trace_rows, 4);
        trace.row_mut(num_rows - 1)[3] = F::ZERO; // So that column 2 and 3 are permutation of one another.
        trace
    }
}

impl Default for FibonacciAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<AP: AirParser> RAir<AP> for FibonacciAir {
    fn eval(&self, parser: &mut AP) {
        // Check public inputs.
        let pis_constraints = [
            parser.sub(parser.local_slice()[0], parser.public_slice()[0]),
            parser.sub(parser.local_slice()[1], parser.public_slice()[1]),
            parser.sub(parser.local_slice()[1], parser.public_slice()[2]),
        ];
        parser.constraint_first_row(pis_constraints[0]);
        parser.constraint_first_row(pis_constraints[1]);
        parser.constraint_last_row(pis_constraints[2]);

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

    fn constraint_degree(&self) -> usize {
        2
    }

    fn width(&self) -> usize {
        4
    }

    fn round_lengths(&self) -> Vec<usize> {
        vec![RAir::<AP>::width(self)]
    }

    fn num_challenges(&self, _round: usize) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::air::parser::TraceWindowParser;

    #[test]
    fn test_fibonacci_air() {
        type F = GoldilocksField;

        let num_rows = 1 << 5 as usize;
        let air = FibonacciAir::new();

        let public_inputs = [
            F::ZERO,
            F::ONE,
            FibonacciAir::fibonacci(num_rows - 1, F::ZERO, F::ONE),
        ];
        let trace = FibonacciAir::generate_trace(F::ZERO, F::ONE, num_rows);

        for window in trace.windows_iter() {
            assert_eq!(window.local_slice.len(), 4);
            let mut window_parser = TraceWindowParser::new(window, &[], &public_inputs);
            assert_eq!(window_parser.local_slice().len(), 4);
            air.eval(&mut window_parser);
        }
    }
}
