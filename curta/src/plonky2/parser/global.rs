use std::collections::HashMap;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::math::prelude::cubic::element::CubicElement;
use crate::plonky2::cubic::builder::CubicCircuitBuilder;
use crate::plonky2::cubic::operations::CubicOperation;
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
    pub(crate) global_vars: &'a [Target],
    pub(crate) public_vars: &'a [Target],
    pub(crate) challenges: &'a [Target],
    pub(crate) cubic_results: &'a mut HashMap<CubicOperation<F>, CubicElement<Target>>,
}

impl<'a, F, FE, P, const D: usize, const D2: usize> AirParser
    for GlobalStarkParser<'a, F, FE, P, D, D2>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    type Field = F;
    type Var = P;

    fn local_slice(&self) -> &[Self::Var] {
        unreachable!("local_slice not implemented for GlobalStarkParser");
    }

    fn next_slice(&self) -> &[Self::Var] {
        unreachable!("next_slice not implemented for GlobalStarkParser");
    }

    fn challenge_slice(&self) -> &[Self::Var] {
        self.challenges
    }

    fn global_slice(&self) -> &[Self::Var] {
        self.global_vars
    }

    fn public_slice(&self) -> &[Self::Var] {
        self.public_vars
    }

    fn constant(&mut self, value: Self::Field) -> Self::Var {
        P::from(FE::from_basefield(value))
    }

    fn constraint(&mut self, constraint: Self::Var) {
        assert_eq!(constraint.as_slice(), P::ZEROS.as_slice());
    }

    fn constraint_transition(&mut self, constraint: Self::Var) {
        assert_eq!(constraint.as_slice(), P::ZEROS.as_slice());
    }

    fn constraint_first_row(&mut self, constraint: Self::Var) {
        assert_eq!(constraint.as_slice(), P::ZEROS.as_slice());
    }

    fn constraint_last_row(&mut self, constraint: Self::Var) {
        assert_eq!(constraint.as_slice(), P::ZEROS.as_slice());
    }

    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        a + b
    }

    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        a - b
    }

    fn neg(&mut self, a: Self::Var) -> Self::Var {
        -a
    }
    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        a * b
    }

    fn zero(&mut self) -> Self::Var {
        P::ZEROS
    }

    fn one(&mut self) -> Self::Var {
        P::ONES
    }

    fn mul_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        a * FE::from_basefield(b)
    }

    fn add_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        a + FE::from_basefield(b)
    }
}

impl<'a, F, FE, P, const D: usize, const D2: usize> PolynomialParser
    for GlobalStarkParser<'a, F, FE, P, D, D2>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
}

impl<'a, F, FE, E: CubicParameters<F>, P, const D: usize, const D2: usize> CubicParser<E>
    for GlobalStarkParser<'a, F, FE, P, D, D2>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
}

impl<'a, F: RichField + Extendable<D>, const D: usize> AirParser
    for GlobalRecursiveStarkParser<'a, F, D>
{
    type Field = F;
    type Var = Target;

    fn constant(&mut self, value: Self::Field) -> Self::Var {
        self.builder.constant(value)
    }

    fn local_slice(&self) -> &[Self::Var] {
        unreachable!("local_slice not implemented for GlobalRecursiveStarkParser");
    }

    fn next_slice(&self) -> &[Self::Var] {
        unreachable!("next_slice not implemented for GlobalRecursiveStarkParser");
    }

    fn challenge_slice(&self) -> &[Self::Var] {
        self.challenges
    }

    fn global_slice(&self) -> &[Self::Var] {
        self.global_vars
    }

    fn public_slice(&self) -> &[Self::Var] {
        self.public_vars
    }

    fn constraint(&mut self, constraint: Self::Var) {
        self.builder.assert_zero(constraint);
    }

    fn constraint_transition(&mut self, constraint: Self::Var) {
        self.builder.assert_zero(constraint);
    }

    fn constraint_first_row(&mut self, constraint: Self::Var) {
        self.builder.assert_zero(constraint);
    }

    fn constraint_last_row(&mut self, constraint: Self::Var) {
        self.builder.assert_zero(constraint);
    }

    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.builder.add(a, b)
    }

    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.builder.sub(a, b)
    }

    fn neg(&mut self, a: Self::Var) -> Self::Var {
        let zero = self.zero();
        self.builder.sub(zero, a)
    }

    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.builder.mul(a, b)
    }

    fn zero(&mut self) -> Self::Var {
        self.builder.zero()
    }

    fn one(&mut self) -> Self::Var {
        self.builder.one()
    }

    fn mul_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        self.builder.mul_const(b, a)
    }
}

impl<'a, F: RichField + Extendable<D>, const D: usize> PolynomialParser
    for GlobalRecursiveStarkParser<'a, F, D>
{
}

impl<'a, F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> CubicParser<E>
    for GlobalRecursiveStarkParser<'a, F, D>
{
    fn add_extension(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) -> CubicElement<Self::Var> {
        self.builder.add_cubic(a, b, self.cubic_results)
    }

    fn sub_extension(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) -> CubicElement<Self::Var> {
        self.builder.sub_cubic(a, b, self.cubic_results)
    }

    fn mul_extension(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) -> CubicElement<Self::Var> {
        self.builder.mul_cubic(a, b, self.cubic_results)
    }

    fn scalar_mul_extension(
        &mut self,
        a: CubicElement<Self::Var>,
        scalar: Self::Var,
    ) -> CubicElement<Self::Var> {
        self.builder.scalar_mul_cubic(a, scalar, self.cubic_results)
    }
}
