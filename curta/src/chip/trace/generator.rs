use alloc::sync::Arc;

use anyhow::{Error, Result};

use super::writer::TraceWriter;
use crate::chip::register::RegisterSerializable;
use crate::chip::table::lookup::Lookup;
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::field::cubic::extension::CubicExtension;
use crate::trace::generator::TraceGenerator;
use crate::trace::AirTrace;

#[derive(Debug, Clone)]
pub struct ArithmeticGenerator<L: AirParameters> {
    writer: TraceWriter<L::Field>,
}

impl<L: ~const AirParameters> ArithmeticGenerator<L> {
    pub fn new(public_inputs: &[L::Field]) -> Self {
        Self {
            writer: TraceWriter::new_with_value(
                L::num_columns(),
                L::num_rows(),
                L::Field::ZERO,
                public_inputs.to_vec(),
            ),
        }
    }

    pub fn new_writer(&self) -> TraceWriter<L::Field> {
        self.writer.clone()
    }

    pub fn trace_clone(&self) -> AirTrace<L::Field> {
        self.writer.read_trace().unwrap().clone()
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.writer.0)
    }
}

impl<L: AirParameters> TraceGenerator<L::Field, Chip<L>> for ArithmeticGenerator<L> {
    type Error = Error;

    fn generate_round(
        &self,
        air: &Chip<L>,
        round: usize,
        challenges: &[L::Field],
        _public_inputs: &[L::Field],
    ) -> Result<AirTrace<L::Field>> {
        match round {
            0 => {
                let trace = self.trace_clone();
                let execution_trace_values = trace
                    .rows_par()
                    .flat_map(|row| row[..air.execution_trace_length].to_vec())
                    .collect::<Vec<_>>();
                Ok(AirTrace {
                    values: execution_trace_values,
                    width: air.execution_trace_length,
                })
            }
            1 => {
                let num_rows = L::num_rows();

                // Write the range check table
                if let Some(table) = &air.range_table {
                    for i in 0..num_rows {
                        self.writer
                            .write(table, &L::Field::from_canonical_usize(i), i);
                    }
                }

                // Write accumulations
                for acc in air.accumulators.iter() {
                    let mut alphas = vec![];
                    for alpha in acc.challenges.iter() {
                        let (a_idx_0, a_idx_1) = alpha.register().get_range();
                        let alpha = CubicExtension::from_base_slice(&challenges[a_idx_0..a_idx_1]);
                        alphas.push(alpha);
                    }
                    self.writer.write_accumulation(num_rows, acc, &alphas);
                }

                // Write lookup proofs
                for data in air.lookup_data.iter() {
                    match data {
                        Lookup::LogDerivative(data) => {
                            let (b_idx_0, b_idx_1) = data.challenge.register().get_range();
                            let beta =
                                CubicExtension::from_base_slice(&challenges[b_idx_0..b_idx_1]);
                            self.writer.write_log_lookup(num_rows, data, beta);
                        }
                    }
                }

                // Write evaluation proofs
                for eval in air.evaluation_data.iter() {
                    let (b_idx_0, b_idx_1) = eval.beta.register().get_range();
                    let beta = CubicExtension::from_base_slice(&challenges[b_idx_0..b_idx_1]);
                    let mut alphas = vec![];
                    for alpha in eval.alphas.iter() {
                        let (a_idx_0, a_idx_1) = alpha.register().get_range();
                        let alpha = CubicExtension::from_base_slice(&challenges[a_idx_0..a_idx_1]);
                        alphas.push(alpha);
                    }
                    self.writer.write_evaluation(num_rows, eval, beta, &alphas);
                }

                let trace = self.trace_clone();
                let extended_trace_values = trace
                    .rows_par()
                    .flat_map(|row| row[air.execution_trace_length..].to_vec())
                    .collect::<Vec<_>>();
                Ok(AirTrace {
                    values: extended_trace_values,
                    width: L::num_columns() - air.execution_trace_length,
                })
            }
            _ => unreachable!("Chip air IOP only has two rounds"),
        }
    }
}
