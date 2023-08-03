use std::sync::mpsc;

use self::builder_operations::ByteLookupOperations;
use self::multiplicity_data::MultiplicityValues;
use self::table::ByteLookupTable;
use super::operations::NUM_BIT_OPPS;
use crate::chip::builder::AirBuilder;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::bytes::operations::instruction::ByteOperationValue;
use crate::chip::AirParameters;

pub mod builder_operations;
pub mod multiplicity_data;
pub mod table;

pub const NUM_CHALLENGES: usize = 6;

impl<L: AirParameters> AirBuilder<L> {
    pub fn byte_operations(
        &mut self,
    ) -> (ByteLookupOperations<L::Field>, ByteLookupTable<L::Field>) {
        let (tx, rx) = mpsc::channel::<ByteOperationValue<L::Field>>();

        let row_acc_challenges = self.alloc_challenge_array::<CubicRegister>(NUM_CHALLENGES);

        todo!()
    }
}
