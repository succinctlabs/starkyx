use super::pointer::RawPointer;
use super::time::TimeRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::Register;
use crate::chip::AirParameters;

pub trait MemoryValue: Register {
    fn compress<L: AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        ptr: RawPointer,
        time: TimeRegister,
    ) -> CubicRegister;
}
