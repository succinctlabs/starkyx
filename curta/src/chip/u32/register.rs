use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;

#[derive(Debug, Clone, Copy)]
pub enum U32Register {
    Element(ElementRegister),
    BitArray(ArrayRegister<BitRegister>),
}
