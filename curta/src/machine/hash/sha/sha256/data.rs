use crate::chip::memory::pointer::slice::Slice;
use crate::chip::register::element::ElementRegister;
use crate::chip::uint::register::U32Register;

pub struct SHA256Data {
    pub state: Slice<U32Register>,
    pub initial_hash: Slice<U32Register>,
    pub w: Slice<U32Register>,
    pub process_id: ElementRegister,
    pub index: ElementRegister,
}
