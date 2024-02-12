use crate::chip::memory::pointer::slice::Slice;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;

pub struct SHAData<T, const LENGTH: usize> {
    pub public: SHAPublicData<T>,
    pub trace: SHATraceData<LENGTH>,
    pub memory: SHAMemory<T>,
    pub degree: usize,
}

pub struct SHAPublicData<T> {
    pub initial_hash: ArrayRegister<T>,
    pub padded_chunks: Vec<ArrayRegister<T>>,
    pub digest_indices: ArrayRegister<ElementRegister>,
}

pub struct SHATraceData<const LENGTH: usize> {
    pub(crate) is_preprocessing: BitRegister,
    pub(crate) process_id: ElementRegister,
    pub(crate) cycle_end_bit: BitRegister,
    pub index: ElementRegister,
    pub is_dummy: BitRegister,
}

pub struct SHAMemory<T> {
    pub(crate) round_constants: Slice<T>,
    pub(crate) w: Slice<T>,
    pub shift_read_mult: Slice<ElementRegister>,
    pub end_bit: Slice<BitRegister>,
    pub digest_bit: Slice<BitRegister>,
    pub dummy_index: ElementRegister,
}
