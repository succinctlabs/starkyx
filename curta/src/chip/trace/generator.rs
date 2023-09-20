use alloc::sync::Arc;

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};

use super::writer::TraceWriter;
use crate::chip::builder::AirTraceData;
use crate::chip::register::element::ElementRegister;
use crate::chip::table::lookup::Lookup;
use crate::chip::{AirParameters, Chip};
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::trace::generator::TraceGenerator;
use crate::trace::AirTrace;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ArithmeticGenerator<L: AirParameters> {
    pub writer: TraceWriter<L::Field>,
    pub air_data: AirTraceData<L>,
}

impl<L: AirParameters> ArithmeticGenerator<L> {
    pub fn reset(&self) {
        let trace_new = AirTrace::new_with_capacity(L::num_columns(), L::num_rows());
        let global_new = Vec::new();
        let challenges_new = Vec::new();
        let public_new = Vec::new();

        let mut trace = self.writer.0.trace.write().unwrap();
        *trace = trace_new;

        let mut global = self.writer.0.global.write().unwrap();
        *global = global_new;

        let mut challenges = self.writer.0.challenges.write().unwrap();
        *challenges = challenges_new;

        let mut public = self.writer.0.public.write().unwrap();
        *public = public_new;
    }

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

    #[inline]
    pub fn range_fn(element: L::Field) -> usize {
        element.as_canonical_u64() as usize
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
                if let Some(Lookup::Element(lookup_data)) = &self.air_data.range_data {
                    let (table_data, values_data) =
                        (&lookup_data.table_data, &lookup_data.values_data);
                    assert_eq!(table_data.table.len(), 1);
                    let table = table_data.table[0];
                    for i in 0..num_rows {
                        self.writer
                            .write(&table, &L::Field::from_canonical_usize(i), i);
                    }

                    self.writer
                        .write_multiplicities_from_fn::<L::CubicParams, ElementRegister>(
                            num_rows,
                            table_data,
                            Self::range_fn,
                            &values_data.trace_values,
                            &values_data.public_values,
                        );
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
