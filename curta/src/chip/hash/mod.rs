use plonky2::iop::target::Target;

pub mod blake;
pub mod sha;

#[derive(Debug, Clone, Copy)]
pub struct CurtaBytes<const N: usize>(pub [Target; N]);
