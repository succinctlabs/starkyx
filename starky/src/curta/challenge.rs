use plonky2::iop::target::Target;

use super::extension::cubic::element::CubicElement;

// #[derive(Debug, Clone, PartialEq, Eq)]
// pub struct StarkChallenges<'a, F>(CubicElement<F>);

pub trait StarkChallenge {
    type Element;
    type ElementPacked;
    type ElementTarget;
}

pub struct CubicChallenge<F>(core::marker::PhantomData<F>);

impl<F> StarkChallenge for CubicChallenge<F> {
    type Element = CubicElement<F>;
    type ElementPacked = CubicElement<F>;
    type ElementTarget = CubicElement<F>;
}
