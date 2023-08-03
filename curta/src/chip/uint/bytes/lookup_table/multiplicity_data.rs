use alloc::collections::BTreeMap;

use crate::chip::uint::bytes::operations::{ByteOperation, NUM_BIT_OPPS};

#[derive(Debug, Clone)]
pub struct MultiplicityData {
    multiplicities_values: Vec<[u32; NUM_BIT_OPPS]>,
    operations_dict: BTreeMap<ByteOperation, (usize, usize)>,
}
