use crate::chip::memory::pointer::slice::Slice;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::register::U32Register;

pub struct SHA256Data {
    pub state: Slice<U32Register>,
    pub initial_hash: ArrayRegister<U32Register>,
    pub round_constants: Slice<U32Register>,
    pub w: Slice<U32Register>,
    pub index: ElementRegister,
    pub is_preprocessing: BitRegister,
    pub process_id: ElementRegister,
    pub dummy_index: ElementRegister,
    pub padded_messages: Vec<ArrayRegister<U32Register>>,
    pub shift_read_mult: Slice<ElementRegister>,
}
