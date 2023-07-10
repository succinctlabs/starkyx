use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::util::transpose;

use crate::curta::stark::Stark;

pub trait AirTraceGenerator<
    S: Stark<F, D, R>,
    F: RichField + Extendable<D>,
    const D: usize,
    const R: usize,
>
{
    fn generate_round(
        &mut self,
        stark: &S,
        round: usize,
        challenges: &[F],
    ) -> Vec<PolynomialValues<F>>;
}

#[derive(Debug, Clone)]
pub struct ConstantGenerator<F: RichField> {
    trace: Vec<PolynomialValues<F>>,
}

impl<F: RichField> ConstantGenerator<F> {
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

impl<S: Stark<F, D, 1>, F: RichField + Extendable<D>, const D: usize> AirTraceGenerator<S, F, D, 1>
    for ConstantGenerator<F>
{
    fn generate_round(
        &mut self,
        _stark: &S,
        round: usize,
        _challenges: &[F],
    ) -> Vec<PolynomialValues<F>> {
        match round {
            0 => self.trace.clone(),
            _ => unreachable!("Round out of bounds"),
        }
    }
}
