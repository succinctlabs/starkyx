use crate::chip::memory::pointer::slice::Slice;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::register::U64Register;

pub struct SHA512Data {
    pub public: SHA512PublicData,
    pub trace: SHA512TraceData,
    pub memory: SHA512Memory,
    pub num_chunks: usize,
    pub degree: usize,
}

pub struct SHA512PublicData {
    pub initial_hash: ArrayRegister<U64Register>,
    pub padded_chunks: Vec<ArrayRegister<U64Register>>,
    pub end_bits: ArrayRegister<BitRegister>,
}

pub struct SHA512TraceData {
    pub(crate) is_preprocessing: BitRegister,
    pub(crate) process_id: ElementRegister,
    pub(crate) cycle_80_end_bit: BitRegister,
    pub index: ElementRegister,
    pub is_dummy: BitRegister,
}

pub struct SHA512Memory {
    pub(crate) round_constants: Slice<U64Register>,
    pub(crate) w: Slice<U64Register>,
    pub shift_read_mult: Slice<ElementRegister>,
    pub end_bit: Slice<BitRegister>,
    pub dummy_index: ElementRegister,
}
