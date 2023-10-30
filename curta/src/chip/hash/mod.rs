use plonky2::iop::target::Target;

pub mod blake;

#[derive(Debug, Clone, Copy)]
pub struct CurtaBytes<const N: usize>(pub [Target; N]);
