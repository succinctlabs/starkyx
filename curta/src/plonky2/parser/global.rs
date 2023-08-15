use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::polynomial::parser::PolynomialParser;

#[derive(Debug, Clone)]
pub struct GlobalStarkParser<'a, F, FE, P, const D: usize, const D2: usize>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    pub(crate) global_vars: &'a [P],
    pub(crate) public_vars: &'a [P],
    pub(crate) challenges: &'a [P],
}

pub struct GlobalRecursiveStarkParser<'a, F: RichField + Extendable<D>, const D: usize> {
    pub(crate) builder: &'a mut CircuitBuilder<F, D>,
    pub(crate) global_vars: &'a [ExtensionTarget<D>],
    pub(crate) public_vars: &'a [Target],
    pub(crate) challenges: &'a [Target],
}