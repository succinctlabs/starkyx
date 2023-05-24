use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;

pub trait TraceGenerator<F: Field, const R: usize> {
    fn generate_round(
        &self,
        round: usize,
        current_values: &[PolynomialValues<F>],
        write_slice: &mut [PolynomialValues<F>],
    );
}

pub struct StarkTrace<F: Field, const R: usize> {
    traces: Vec<PolynomialValues<F>>,
    round_indices: [(usize, usize); R],
}

impl<F: Field, const R: usize> StarkTrace<F, R> {
    pub fn new(num_columns: usize) -> Self {
        Self {
            traces: Vec::with_capacity(num_columns),
            round_indices: [(0, 0); R],
        }
    }

    pub fn split_round(
        &mut self,
        round: usize,
    ) -> (&[PolynomialValues<F>], &mut [PolynomialValues<F>]) {
        let (read_slice, write_slice) = self.traces.split_at_mut(self.round_indices[round].0);
        (read_slice, &mut write_slice[..self.round_indices[round].1])
    }
}
