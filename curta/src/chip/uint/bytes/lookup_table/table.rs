use core::array::from_fn;
use std::sync::mpsc::Receiver;

use super::super::operations::NUM_BIT_OPPS;
use super::multiplicity_data::MultiplicityData;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::register::ByteRegister;
use crate::chip::AirParameters;

#[derive(Debug)]
pub struct ByteLookupTable<F> {
    pub a: ByteRegister,
    pub b: ByteRegister,
    pub results: [ByteRegister; NUM_BIT_OPPS],
    a_bits: ArrayRegister<BitRegister>,
    b_bits: ArrayRegister<BitRegister>,
    results_bits: [ArrayRegister<BitRegister>; NUM_BIT_OPPS],
    multiplicity_data: MultiplicityData<F>,
    row_acc_challenges: ArrayRegister<CubicRegister>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_byte_lookup_table(
        &mut self,
        row_acc_challenges: ArrayRegister<CubicRegister>,
        rx: Receiver<ByteOperation<L::Field>>,
    ) -> ByteLookupTable<L::Field> {
        let multiplicities = self.alloc_array::<ElementRegister>(NUM_BIT_OPPS);

        let a = self.alloc::<ByteRegister>();
        let b = self.alloc::<ByteRegister>();
        let results = from_fn::<_, NUM_BIT_OPPS, _>(|_| self.alloc::<ByteRegister>());

        let a_bits = self.alloc_array::<BitRegister>(8);
        let b_bits = self.alloc_array::<BitRegister>(8);
        let results_bits = from_fn::<_, NUM_BIT_OPPS, _>(|_| self.alloc_array::<BitRegister>(8));

        let multiplicity_data = MultiplicityData::new(L::num_rows(), rx, multiplicities);

        ByteLookupTable {
            a,
            b,
            results,
            a_bits,
            b_bits,
            results_bits,
            multiplicity_data,
            row_acc_challenges,
        }
    }
}
