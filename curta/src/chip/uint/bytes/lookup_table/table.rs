use std::sync::mpsc::Receiver;

use super::super::operations::{ByteOperation, NUM_BIT_OPPS, NUM_OUTPUT_CARRY_BITS};
use super::builder_operations::ByteLookupOperations;
use super::multiplicity_data::MultiplicityData;
use super::NUM_CHALLENGES;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::bytes::operations::instruction::ByteOperationValue;
use crate::chip::uint::bytes::operations::NUM_INPUT_CARRY_BITS;
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
    result_carry_bits: [BitRegister; NUM_OUTPUT_CARRY_BITS],
    input_carry_bits: [BitRegister; NUM_INPUT_CARRY_BITS],
    multiplicity_data: MultiplicityData<F>,
    row_acc_challenges: ArrayRegister<CubicRegister>,
    row_digests: [CubicRegister; NUM_BIT_OPPS],
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn new_byte_lookup_table(
        &mut self,
        row_acc_challenges: ArrayRegister<CubicRegister>,
        rx: Receiver<ByteOperationValue<L::Field>>,
    ) -> ByteLookupTable<L::Field> {
        let multiplicities = self.alloc_array::<ElementRegister>(NUM_BIT_OPPS);

        let a = self.alloc::<ByteRegister>();
        let b = self.alloc::<ByteRegister>();
        let results = [self.alloc::<ByteRegister>(); NUM_BIT_OPPS];

        let a_bits = self.alloc_array::<BitRegister>(8);
        let b_bits = self.alloc_array::<BitRegister>(8);
        let results_bits = [self.alloc_array::<BitRegister>(8); NUM_BIT_OPPS];
        let result_carry_bits = [self.alloc::<BitRegister>(); NUM_OUTPUT_CARRY_BITS];
        let input_carry_bits = [self.alloc::<BitRegister>(); NUM_INPUT_CARRY_BITS];

        todo!()
    }
}
