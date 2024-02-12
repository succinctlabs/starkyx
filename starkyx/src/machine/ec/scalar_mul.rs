use crate::chip::ec::EllipticCurve;
use crate::chip::field::register::FieldRegister;
use crate::chip::memory::pointer::slice::Slice;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;

pub struct DoubleAddData<E: EllipticCurve> {
    pub process_id: ElementRegister,
    pub temp_x_ptr: Slice<FieldRegister<E::BaseField>>,
    pub temp_y_ptr: Slice<FieldRegister<E::BaseField>>,
    pub bit: BitRegister,
    pub start_bit: BitRegister,
    pub end_bit: BitRegister,
}
