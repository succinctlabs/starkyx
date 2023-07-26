use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;

#[derive(Debug, Clone)]
pub struct And {
    pub a: ElementRegister,
    pub b: ElementRegister,
    pub result: ElementRegister,
    a_bits: ArrayRegister<BitRegister>,
    b_bits: ArrayRegister<BitRegister>,
    result_bits: ArrayRegister<BitRegister>,
}
