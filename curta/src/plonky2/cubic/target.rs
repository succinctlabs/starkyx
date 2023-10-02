use core::ops::Range;

use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::GeneratedValues;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};

use crate::math::prelude::cubic::element::CubicElement;

impl CubicElement<Target> {
    pub fn from_range(row: usize, range: Range<usize>) -> Self {
        debug_assert_eq!(range.end - range.start, 3);
        CubicElement([
            Target::wire(row, range.start),
            Target::wire(row, range.start + 1),
            Target::wire(row, range.start + 2),
        ])
    }

    pub fn get<F: RichField>(&self, witness: &PartitionWitness<F>) -> CubicElement<F> {
        CubicElement(witness.get_targets(&self.0).try_into().unwrap())
    }

    pub fn set<F: RichField>(&self, value: &CubicElement<F>, out_buffer: &mut GeneratedValues<F>) {
        out_buffer.set_target_arr(&self.0, &value.0);
    }
}
