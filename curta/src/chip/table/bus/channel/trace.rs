use super::entry::Entry;
use super::BusChannel;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::RegisterSerializable;
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
                let filter = CubicElement::from_slice(&filter_vec[0..3]);
                if filter == CubicElement::ZERO {
                    CubicExtension::ONE
                } else {
                    beta - CubicExtension::<F, E>::from(self.read(value, row_index))
                }
            }
            Entry::Output(value, filter) => {
                let filter_vec = self.read_expression(filter, row_index);
                assert_eq!(filter_vec.len(), 1);
                let filter = CubicElement::from_slice(&filter_vec[0..3]);
                if filter == CubicElement::ZERO {
                    CubicExtension::ONE
                } else {
                    let value = beta - CubicExtension::<F, E>::from(self.read(value, row_index));
                    value.inverse()
                }
            }
        }
    }

    pub fn write_bus_channel<E: CubicParameters<F>>(
        &self,
        num_rows: usize,
        channel: &BusChannel<F, E>,
    ) {
        let beta = CubicExtension::<F, E>::from(self.read(&channel.challenge, 0));

        // Calculate the bus values
        let bus_values = (0..num_rows)
            .scan(CubicExtension::<F, E>::ONE, |acc, i| {
                Some(*acc * channel.entries.iter().map(|entry|
                    self.read_entry(entry, beta, i) 
                ).product::<CubicExtension<F, E>>())
            })
            .collect::<Vec<_>>();
        assert_eq!(bus_values.len(), num_rows);

        //  Write the bus values to the trace
        todo!();
    }
}
