use itertools::Itertools;

use super::entry::Entry;
use super::BusChannel;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::maybe_rayon::*;
use crate::plonky2::field::cubic::element::CubicElement;
use crate::plonky2::field::cubic::extension::CubicExtension;

impl<F: PrimeField> TraceWriter<F> {
    pub fn read_entry<E: CubicParameters<F>>(
        &self,
        entry: &Entry<F>,
        beta: CubicExtension<F, E>,
        row_index: usize,
    ) -> CubicExtension<F, E> {
        match entry {
            Entry::Input(value, filter) => {
                let filter_vec = self.read_expression(filter, row_index);
                assert_eq!(filter_vec.len(), 1);
                let filter = CubicExtension::from(CubicElement::from_slice(&filter_vec[0..3]));
                let value = beta - CubicExtension::from(self.read(value, row_index));
                let one = CubicExtension::<F, E>::ONE;
                filter * value + (one - filter) * one
            }
            Entry::Output(value, filter) => {
                let filter_vec = self.read_expression(filter, row_index);
                assert_eq!(filter_vec.len(), 1);
                let filter = CubicExtension::from(CubicElement::from_slice(&filter_vec[0..3]));
                let value = (beta - CubicExtension::from(self.read(value, row_index))).inverse();
                let one = CubicExtension::<F, E>::ONE;
                filter * value + (one - filter) * one
            }
        }
    }

    pub fn write_bus_channel<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        channel: &BusChannel<F, E>,
    ) {
        let beta = CubicExtension::<F, E>::from(self.read(&channel.challenge, 0));

        // Calculate the accumulated values,
        // assigning the entry values to the trace as we go
        let accumulated_values = (0..num_rows)
            .scan(CubicExtension::<F, E>::ONE, |acc, i| {
                Some(
                    *acc * channel
                        .entries
                        .iter()
                        .zip_eq(channel.entry_values.iter())
                        .map(|(entry, register)| {
                            let entry_value = self.read_entry(entry, beta, i);
                            self.write(register, &entry_value.0, i);
                            entry_value
                        })
                        .product::<CubicExtension<F, E>>(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(accumulated_values.len(), num_rows);

        //  Write the accumulating bus table values to the trace
        self.write_trace()
            .unwrap()
            .rows_par_mut()
            .zip(accumulated_values.par_iter())
            .for_each(|(row, value)| channel.table_accumulator.assign_to_raw_slice(row, &value.0));

        // Write the row-wise running product value
        let product_chunks = channel.entry_values.chunks_exact(2);
        self.write_trace().unwrap().rows_par_mut().for_each(|row| {
            let chunks = product_chunks.clone();
            let mut acc = CubicExtension::<F, E>::ONE;
            for (chunk, acc_reg) in chunks.zip(channel.row_acc_product.iter()) {
                let a = CubicExtension::<F, E>::from(chunk[0].read_from_slice(row));
                let b = CubicExtension::<F, E>::from(chunk[1].read_from_slice(row));
                acc *= a * b;
                acc_reg.assign_to_raw_slice(row, &acc.0);
            }
        });

        // Write the final accumulated value to the output channel
        self.write_global(&channel.out_channel, &accumulated_values[num_rows - 1].0)
    }
}
