use std::sync::mpsc;

use self::builder_operations::ByteLookupOperations;
use self::table::ByteLookupTable;
use crate::chip::builder::AirBuilder;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::uint::bytes::operations::value::ByteOperation;
use crate::chip::uint::bytes::operations::NUM_CHALLENGES;
use crate::chip::AirParameters;

pub mod builder_operations;
pub mod multiplicity_data;
pub mod table;

impl<L: AirParameters> AirBuilder<L> {
    pub fn byte_operations(
        &mut self,
    ) -> (ByteLookupOperations<L::Field>, ByteLookupTable<L::Field>) {
        let (tx, rx) = mpsc::channel::<ByteOperation<L::Field>>();

        let row_acc_challenges = self.alloc_challenge_array::<CubicRegister>(NUM_CHALLENGES);

        let lookup_table = self.new_byte_lookup_table(row_acc_challenges, rx);
        let operations = ByteLookupOperations::new(tx, row_acc_challenges);

        (operations, lookup_table)
    }
}
