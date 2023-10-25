use crate::chip::memory::pointer::slice::Slice;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::register::U32Register;

pub struct SHA256Data {
    pub public: SHA256PublicData,
    pub trace: SHA256TraceData,
    pub memory: SHA256Memory,
    pub num_chunks: usize,
}

pub struct SHA256PublicData {
    pub initial_hash: ArrayRegister<U32Register>,
    pub padded_chunks: Vec<ArrayRegister<U32Register>>,
    pub end_bits: ArrayRegister<BitRegister>,
}

pub struct SHA256TraceData {
    pub(crate) is_preprocessing: BitRegister,
    pub(crate) process_id: ElementRegister,
    pub(crate) cycle_64_end_bit: BitRegister,
    pub index: ElementRegister,
}

pub struct SHA256Memory {
    pub(crate) round_constants: Slice<U32Register>,
    pub(crate) w: Slice<U32Register>,
    pub shift_read_mult: Slice<ElementRegister>,
    pub end_bit: Slice<BitRegister>,
    pub dummy_index: ElementRegister,
}
