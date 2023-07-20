use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, Hasher};

use super::parser::{RecursiveStarkParser, StarkParser};
use crate::challenger::Challenger;

pub struct Plonky2Challenger<F: RichField, H: Hasher<F>>(pub challenger::Challenger<F, H>);

impl<F: RichField, H: Hasher<F>> Plonky2Challenger<F, H> {
    pub fn new() -> Self {
        Self(challenger::Challenger::new())
    }
}

impl<'a, F, FE, P, const D: usize, const D2: usize, H: Hasher<F>>
    Challenger<StarkParser<'a, F, FE, P, D, D2>> for Plonky2Challenger<F, H>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    type Element = F;

    fn observe_element(&mut self, _parser: &mut StarkParser<'a, F, FE, P, D, D2>, element: &F) {
        self.0.observe_element(*element)
    }

    fn challenge(&mut self, _parser: &mut StarkParser<'a, F, FE, P, D, D2>) -> P {
        FE::from_basefield(self.0.get_challenge()).into()
    }
}

pub struct Plonky2RecursiveChallenger<
    F: RichField + Extendable<D>,
    H: AlgebraicHasher<F>,
    const D: usize,
>(pub challenger::RecursiveChallenger<F, H, D>);

impl<F: RichField + Extendable<D>, H: AlgebraicHasher<F>, const D: usize>
    Plonky2RecursiveChallenger<F, H, D>
{
    pub fn new(builder: &mut CircuitBuilder<F, D>) -> Self {
        Self(challenger::RecursiveChallenger::new(builder))
    }
}

impl<'a, F: RichField + Extendable<D>, H: AlgebraicHasher<F>, const D: usize>
    Challenger<RecursiveStarkParser<'a, F, D>> for Plonky2RecursiveChallenger<F, H, D>
{
    type Element = Target;

    fn observe_element(&mut self, _parser: &mut RecursiveStarkParser<'a, F, D>, element: &Target) {
        self.0.observe_elements(&[*element])
    }

    fn challenge(&mut self, parser: &mut RecursiveStarkParser<'a, F, D>) -> ExtensionTarget<D> {
        let challenge_target = self.0.get_challenge(parser.builder);
        parser.builder.convert_to_ext(challenge_target)
    }
}
