use crate::chip::memory::pointer::slice::Slice;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::register::U64Register;

pub struct BLAKE2BData {
    pub public: BLAKE2BPublicData,
    pub trace: BLAKE2BTraceData,
    pub memory: BLAKE2BMemory,
    pub num_chunks: usize,
}

pub struct BLAKE2BPublicData {
    pub initial_hash: ArrayRegister<U64Register>,
    pub initial_hash_compress: ArrayRegister<U64Register>,
    pub padded_chunks: Vec<ArrayRegister<U64Register>>,
    pub end_bits: ArrayRegister<BitRegister>,
}

pub struct BLAKE2BTraceData {
    pub(crate) is_compress_initialize: BitRegister,
    pub(crate) process_id: ElementRegister,
    pub(crate) cycle_8_end_bit: BitRegister, // Used for each mix iteration
    pub(crate) cycle_96_end_bit: BitRegister, // Used for each compress round
    pub(crate) mix_iteration: ElementRegister,
    pub(crate) mix_index: ElementRegister,
}

pub struct BLAKE2BMemory {
    pub(crate) permutations: Vec<Slice<ElementRegister>>,
    pub(crate) v: Slice<U64Register>,
    pub(crate) m: Slice<U64Register>,
    pub(crate) end_bit: Slice<BitRegister>,
}
