use super::pointer::raw::RawPointer;
use super::time::Time;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::Register;
use crate::chip::AirParameters;

pub trait MemoryValue: Register {
    fn num_challenges() -> usize;

    fn compress<L: AirParameters>(
        &self,
        builder: &mut AirBuilder<L>,
        ptr: RawPointer,
        time: &Time<L::Field>,
        challenges: &ArrayRegister<CubicRegister>,
    ) -> CubicRegister;
}
