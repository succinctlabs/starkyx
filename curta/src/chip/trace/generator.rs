use alloc::sync::Arc;

use anyhow::{Error, Result};

use super::writer::TraceWriter;
use crate::chip::builder::AirTraceData;
use crate::chip::table::lookup::Lookup;
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::trace::generator::TraceGenerator;
use crate::trace::AirTrace;

#[derive(Debug, Clone)]
pub struct ArithmeticGenerator<L: AirParameters> {
    writer: TraceWriter<L::Field>,
    pub air_data: AirTraceData<L>,
}

impl<L: ~const AirParameters> ArithmeticGenerator<L> {
    // pub fn new(air: &Chip<L>) -> Self {
    //     let num_public_inputs = air.num_public_inputs;
    //     let num_global_values = air.num_global_values;
    //      Self {
    //         writer: TraceWriter::new_with_value(
    //             L::Field::ZERO,
    //             L::num_columns(),
    //             L::num_rows(),
    //             num_public_inputs,
    //             num_global_values,
    //         ),
    //     }
    // }

    pub fn new(air_data: AirTraceData<L>) -> Self {
        let num_public_inputs = air_data.num_public_inputs;
        let num_global_values = air_data.num_global_values;
        Self {
            writer: TraceWriter::new_with_value(
                L::Field::ZERO,
                L::num_columns(),
                L::num_rows(),
                num_public_inputs,
                num_global_values,
            ),
            air_data,
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
        global_values: &mut [L::Field],
        public_inputs: &[L::Field],
    ) -> Result<AirTrace<L::Field>> {
        match round {
            0 => {
                let (id_0, id_1) = (0, self.air_data.num_public_inputs);
                let mut public_write = self.writer.0.public.write().unwrap();
                public_write[id_0..id_1].copy_from_slice(&public_inputs[id_0..id_1]);
                drop(public_write);

                let num_rows = L::num_rows();

                // Write the range check table and multiplicitiies
                if let Some(table) = &self.air_data.range_table {
                    for i in 0..num_rows {
                        self.writer
                            .write(table, &L::Field::from_canonical_usize(i), i);
                    }
                }

                // Write multiplicities for lookup table with search functions
                for data in self.air_data.lookup_data.iter() {
                    if let Lookup::LogDerivative(log_data) = data {
                        if let Some(table_index) = log_data.table_index {
                            self.writer.write_multiplicities_from_fn(
                                num_rows,
                                &log_data.table_data,
                                table_index,
                                &log_data.values_data.values,
                            );
                        }
                    }
                }

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

                // Insert the challenges into the generator
                let writer = self.new_writer();
                let mut challenges_write = writer.challenges.write().unwrap();
                challenges_write.extend_from_slice(challenges);
                drop(challenges_write);

                // Write accumulations
                for acc in self.air_data.accumulators.iter() {
                    self.writer.write_accumulation(acc);
                }

                for channel in self.air_data.bus_channels.iter() {
                    self.writer.write_bus_channel(num_rows, channel);
                }

                // Write lookup proofs
                for data in self.air_data.lookup_data.iter() {
                    self.writer.write_lookup(num_rows, data);
                }

                // Write evaluation proofs
                for eval in self.air_data.evaluation_data.iter() {
                    self.writer.write_evaluation(num_rows, eval);
                }

                let trace = self.trace_clone();
                let extended_trace_values = trace
                    .rows_par()
                    .flat_map(|row| row[air.execution_trace_length..].to_vec())
                    .collect::<Vec<_>>();

                let new_global = self.writer.0.global.read().unwrap();
                let (id_0, id_1) = (0, air.num_global_values);
                global_values[id_0..id_1].copy_from_slice(&new_global[id_0..id_1]);
                drop(new_global);
                Ok(AirTrace {
                    values: extended_trace_values,
                    width: L::num_columns() - air.execution_trace_length,
                })
            }
            _ => unreachable!("Chip air IOP only has two rounds"),
        }
    }
}
