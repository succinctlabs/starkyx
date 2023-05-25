use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::util::transpose;
use plonky2_maybe_rayon::*;

pub trait TraceGenerator<F: Field, const R: usize> {
    fn generate_round(
        &self,
        round: usize,
        challenges: &[F],
        current_values: &[PolynomialValues<F>],
        write_slice: &mut [PolynomialValues<F>],
    );
}

pub struct StarkTrace<F: Field, const R: usize> {
    trace: Vec<PolynomialValues<F>>,
    round_indices: [(usize, usize); R],
}

impl<F: Field, const R: usize> StarkTrace<F, R> {
    pub fn new(num_rows: usize, round_indices: [(usize, usize); R]) -> Self {
        let trace = vec![
            PolynomialValues::new(vec![F::ZERO; num_rows]);
            round_indices[R - 1].0 + round_indices[R - 1].1
        ];
        Self {
            trace,
            round_indices,
        }
    }

    pub fn split_round(
        &mut self,
        round: usize,
    ) -> (&[PolynomialValues<F>], &mut [PolynomialValues<F>]) {
        let (read_slice, write_slice) = self.trace.split_at_mut(self.round_indices[round].0);
        (read_slice, &mut write_slice[..self.round_indices[round].1])
    }
}

#[derive(Debug, Clone)]
pub struct ConstantGenerator<F: Field> {
    trace: Vec<PolynomialValues<F>>,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn new(trace: Vec<PolynomialValues<F>>) -> Self {
        Self { trace }
    }

    pub fn from_cols(trace_cols: Vec<Vec<F>>) -> Self {
        Self::new(trace_cols.into_iter().map(PolynomialValues::new).collect())
    }

    pub fn from_rows(trace_rows: Vec<Vec<F>>) -> Self {
        let trace_cols = transpose(&trace_rows);
        Self::from_cols(trace_cols)
    }
}

impl<F: Field, const R: usize> TraceGenerator<F, R> for ConstantGenerator<F> {
    fn generate_round(
        &self,
        _round: usize,
        _challenges: &[F],
        _current_values: &[PolynomialValues<F>],
        write_slice: &mut [PolynomialValues<F>],
    ) {
        write_slice
            .par_iter_mut()
            .zip(self.trace.par_iter())
            .for_each(|(col, val)| col.values.copy_from_slice(&val.values));
    }
}
