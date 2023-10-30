use std::collections::HashMap;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::operations::{CubicBuilderOperations, CubicOperation};
use crate::math::prelude::cubic::element::CubicElement;

pub trait CubicCircuitBuilder<F: RichField + Extendable<D>, const D: usize> {
    fn zero_cubic(&mut self) -> CubicElement<Target>;

    fn one_cubic(&mut self) -> CubicElement<Target>;

    fn cubic_from_base(&mut self, value: Target) -> CubicElement<Target>;

    fn arithmetic_cubic(
        &mut self,
        const_0: F,
        const_1: F,
        multiplicand_0: CubicElement<Target>,
        multiplicand_1: CubicElement<Target>,
        addend: CubicElement<Target>,
        cubic_results: &mut HashMap<CubicOperation<F>, CubicElement<Target>>,
    ) -> CubicElement<Target>;

    fn add_cubic(
        &mut self,
        a: CubicElement<Target>,
        b: CubicElement<Target>,
        cubic_results: &mut HashMap<CubicOperation<F>, CubicElement<Target>>,
    ) -> CubicElement<Target> {
        let one = self.one_cubic();
        self.arithmetic_cubic(F::ONE, F::ONE, one, a, b, cubic_results)
    }

    fn sub_cubic(
        &mut self,
        a: CubicElement<Target>,
        b: CubicElement<Target>,
        cubic_results: &mut HashMap<CubicOperation<F>, CubicElement<Target>>,
    ) -> CubicElement<Target> {
        let one = self.one_cubic();
        self.arithmetic_cubic(F::ONE, -F::ONE, one, a, b, cubic_results)
    }

    fn mul_cubic(
        &mut self,
        a: CubicElement<Target>,
        b: CubicElement<Target>,
        cubic_results: &mut HashMap<CubicOperation<F>, CubicElement<Target>>,
    ) -> CubicElement<Target> {
        let zero = self.zero_cubic();
        self.arithmetic_cubic(F::ONE, F::ZERO, a, b, zero, cubic_results)
    }

    fn scalar_mul_cubic(
        &mut self,
        a: CubicElement<Target>,
        b: Target,
        cubic_results: &mut HashMap<CubicOperation<F>, CubicElement<Target>>,
    ) -> CubicElement<Target> {
        let b_ext = self.cubic_from_base(b);
        self.mul_cubic(a, b_ext, cubic_results)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> CubicCircuitBuilder<F, D>
    for CircuitBuilder<F, D>
{
    fn zero_cubic(&mut self) -> CubicElement<Target> {
        CubicBuilderOperations::zero(self)
    }

    fn one_cubic(&mut self) -> CubicElement<Target> {
        CubicElement([self.one(), self.zero(), self.zero()])
    }

    fn cubic_from_base(&mut self, value: Target) -> CubicElement<Target> {
        CubicElement([value, self.zero(), self.zero()])
    }

    fn arithmetic_cubic(
        &mut self,
        const_0: F,
        const_1: F,
        multiplicand_0: CubicElement<Target>,
        multiplicand_1: CubicElement<Target>,
        addend: CubicElement<Target>,
        cubic_results: &mut HashMap<CubicOperation<F>, CubicElement<Target>>,
    ) -> CubicElement<Target> {
        CubicBuilderOperations::arithmetic_cubic(
            self,
            const_0,
            const_1,
            multiplicand_0,
            multiplicand_1,
            addend,
            cubic_results,
        )
    }
}
