use super::operations::NUM_BIT_OPPS;
use super::register::ByteRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::table::lookup::log_der::{LogLookupValues, LookupTable};
use crate::chip::AirParameters;

pub const NUM_CHALLENGES: usize = 1 + 6 * 3;

#[derive(Debug, Clone)]
pub struct ByteLookupOperations {
    row_acc_challenges: ArrayRegister<CubicRegister>,
    multiplicities: ArrayRegister<ElementRegister>,
    multiplicities_values: [u32; NUM_BIT_OPPS],
    values: Vec<CubicRegister>,
}

#[derive(Debug, Clone, Copy)]
pub struct ByteLookupTable {
    pub a: ByteRegister,
    pub b: ByteRegister,
    pub results: ArrayRegister<ByteRegister>,
    a_bits: ArrayRegister<BitRegister>,
    b_bits: ArrayRegister<BitRegister>,
    opcodes: [ElementRegister; NUM_BIT_OPPS],
    results_bits: [ArrayRegister<BitRegister>; NUM_BIT_OPPS],
    carry_bits: [BitRegister; NUM_BIT_OPPS],
    row_acc_challenges: ArrayRegister<CubicRegister>,
    lookup_challenge: CubicRegister,
    row_digests: [CubicRegister; NUM_BIT_OPPS],
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn bytes_lookup(&mut self) -> (ByteLookupTable, ByteLookupOperations) {
        let a = self.alloc::<ByteRegister>();
        let b = self.alloc::<ByteRegister>();
        let results = self.alloc_array::<ByteRegister>(8);
        let a_bits = self.alloc_array::<BitRegister>(8);
        let b_bits = self.alloc_array::<BitRegister>(8);
        let results_bits = [self.alloc_array::<BitRegister>(8); NUM_BIT_OPPS];
        let carry_bits = [self.alloc::<BitRegister>(); NUM_BIT_OPPS];
        let row_acc_challenges = self.alloc_challenge_array(NUM_CHALLENGES);
        let lookup_challenge = self.alloc_challenge::<CubicRegister>();
        let opcodes = [self.alloc::<ElementRegister>(); NUM_BIT_OPPS];
        let multiplicities = self.alloc_array::<ElementRegister>(NUM_BIT_OPPS);

        // Accumulate operations and opcodes
        let row_digests: [_; NUM_BIT_OPPS] = opcodes
            .iter()
            .zip(results.iter())
            .map(|(opcode, result)| {
                let values = [*opcode, a.element(), b.element(), result.element()];
                self.accumulate(&row_acc_challenges, &values)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        (
            ByteLookupTable {
                a,
                b,
                results,
                a_bits,
                b_bits,
                opcodes,
                results_bits,
                carry_bits,
                row_acc_challenges,
                lookup_challenge,
                row_digests,
            },
            ByteLookupOperations {
                row_acc_challenges,
                multiplicities,
                multiplicities_values: [0; NUM_BIT_OPPS],
                values: Vec::new(),
            },
        )
    }

    pub fn constrain_byte_operations(
        &mut self,
        lookup_table: &ByteLookupTable,
        operations: &ByteLookupOperations,
    ) {
        let challenge = lookup_table.lookup_challenge;

        let lookup_table = self.lookup_table_with_multiplicities(
            &challenge,
            &lookup_table.row_digests,
            &operations.multiplicities,
        );
        let lookup_values = self.lookup_values(&challenge, &operations.values);

        // Constrain the lookup table
        self.cubic_lookup_from_table_and_values(lookup_table, lookup_values);

        // Constrain the byte operations
        todo!();
    }
}
