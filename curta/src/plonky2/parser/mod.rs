pub mod consumer;
pub mod global;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use self::consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::air::extension::cubic::CubicParser;
use crate::air::parser::AirParser;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::polynomial::parser::PolynomialParser;

pub struct StarkParser<'a, F, FE, P, const D: usize, const D2: usize>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    pub(crate) local_vars: &'a [P],
    pub(crate) next_vars: &'a [P],
    pub(crate) global_vars: &'a [P],
    pub(crate) public_vars: &'a [P],
    pub(crate) challenges: &'a [P],
    pub(crate) consumer: &'a mut ConstraintConsumer<P>,
}

pub struct RecursiveStarkParser<'a, F: RichField + Extendable<D>, const D: usize> {
    pub(crate) builder: &'a mut CircuitBuilder<F, D>,
    pub(crate) local_vars: &'a [ExtensionTarget<D>],
    pub(crate) next_vars: &'a [ExtensionTarget<D>],
    pub(crate) global_vars: &'a [ExtensionTarget<D>],
    pub(crate) public_vars: &'a [ExtensionTarget<D>],
    pub(crate) challenges: &'a [ExtensionTarget<D>],
    pub(crate) consumer: &'a mut RecursiveConstraintConsumer<F, D>,
}

impl<'a, F, FE, P, const D: usize, const D2: usize> AirParser for StarkParser<'a, F, FE, P, D, D2>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    type Field = F;
    type Var = P;

    fn local_slice(&self) -> &[Self::Var] {
        self.local_vars
    }

    fn next_slice(&self) -> &[Self::Var] {
        self.next_vars
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
        self.consumer.constraint(constraint);
    }

    fn constraint_transition(&mut self, constraint: Self::Var) {
        self.consumer.constraint_transition(constraint);
    }

    fn constraint_first_row(&mut self, constraint: Self::Var) {
        self.consumer.constraint_first_row(constraint);
    }

    fn constraint_last_row(&mut self, constraint: Self::Var) {
        self.consumer.constraint_last_row(constraint);
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
    for StarkParser<'a, F, FE, P, D, D2>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
}

impl<'a, F, FE, E: CubicParameters<F>, P, const D: usize, const D2: usize> CubicParser<E>
    for StarkParser<'a, F, FE, P, D, D2>
where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
}

impl<'a, F: RichField + Extendable<D>, const D: usize> AirParser
    for RecursiveStarkParser<'a, F, D>
{
    type Field = F;
    type Var = ExtensionTarget<D>;

    fn constant(&mut self, value: Self::Field) -> Self::Var {
        self.builder.constant_extension(F::Extension::from(value))
    }

    fn local_slice(&self) -> &[Self::Var] {
        self.local_vars
    }

    fn next_slice(&self) -> &[Self::Var] {
        self.next_vars
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
        self.consumer.constraint(self.builder, constraint);
    }

    fn constraint_transition(&mut self, constraint: Self::Var) {
        self.consumer
            .constraint_transition(self.builder, constraint);
    }

    fn constraint_first_row(&mut self, constraint: Self::Var) {
        self.consumer.constraint_first_row(self.builder, constraint);
    }

    fn constraint_last_row(&mut self, constraint: Self::Var) {
        self.consumer.constraint_last_row(self.builder, constraint);
    }

    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.builder.add_extension(a, b)
    }

    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.builder.sub_extension(a, b)
    }

    fn neg(&mut self, a: Self::Var) -> Self::Var {
        let zero = self.zero();
        self.builder.sub_extension(zero, a)
    }

    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.builder.mul_extension(a, b)
    }

    fn zero(&mut self) -> Self::Var {
        self.builder.zero_extension()
    }

    fn one(&mut self) -> Self::Var {
        self.builder.one_extension()
    }

    fn mul_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.builder.constant(b);
        self.builder.scalar_mul_ext(b, a)
    }
}

impl<'a, F: RichField + Extendable<D>, const D: usize> PolynomialParser
    for RecursiveStarkParser<'a, F, D>
{
}

impl<'a, F: RichField + Extendable<D>, E: CubicParameters<F>, const D: usize> CubicParser<E>
    for RecursiveStarkParser<'a, F, D>
{
}
