use core::marker::PhantomData;
use std::collections::HashMap;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::arithmetic_gate::ArithmeticCubicGate;
use super::mul_gate::MulCubicGate;
use crate::math::prelude::cubic::element::CubicElement;
use crate::math::prelude::*;

/// Represents an extension arithmetic operation in the circuit. Used to memoize results.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct CubicOperation<F> {
    pub const_0: F,
    pub const_1: F,
    pub multiplicand_0: CubicElement<Target>,
    pub multiplicand_1: CubicElement<Target>,
    pub addend: CubicElement<Target>,
}

pub(crate) struct CubicBuilderOperations<F, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> CubicBuilderOperations<F, D> {
    pub(crate) fn zero(builder: &mut CircuitBuilder<F, D>) -> CubicElement<Target> {
        CubicElement([builder.zero(), builder.zero(), builder.zero()])
    }

    pub(crate) fn cubic_target_as_constant(
        builder: &CircuitBuilder<F, D>,
        target: CubicElement<Target>,
    ) -> Option<CubicElement<F>> {
        let const_coefficients = target
            .0
            .iter()
            .filter_map(|&t| builder.target_as_constant(t))
            .collect::<Vec<_>>();

        if let Ok(coefficients) = const_coefficients.try_into() {
            Some(CubicElement(coefficients))
        } else {
            None
        }
    }

    pub(crate) fn constant(
        builder: &mut CircuitBuilder<F, D>,
        constant: CubicElement<F>,
    ) -> CubicElement<Target> {
        CubicElement([
            builder.constant(constant.0[0]),
            builder.constant(constant.0[1]),
            builder.constant(constant.0[2]),
        ])
    }

    pub(crate) fn connect(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<Target>,
        b: CubicElement<Target>,
    ) {
        for (a, b) in a.0.iter().zip(b.0.iter()) {
            builder.connect(*a, *b);
        }
    }

    pub(crate) fn arithmetic_cubic(
        builder: &mut CircuitBuilder<F, D>,
        const_0: F,
        const_1: F,
        multiplicand_0: CubicElement<Target>,
        multiplicand_1: CubicElement<Target>,
        addend: CubicElement<Target>,
        cubic_results: &mut HashMap<CubicOperation<F>, CubicElement<Target>>,
    ) -> CubicElement<Target> {
        // See if we can determine the result without adding an `ArithmeticGate`.
        if let Some(result) = Self::cubic_extension_special_cases(
            builder,
            const_0,
            const_1,
            multiplicand_0,
            multiplicand_1,
            addend,
        ) {
            return result;
        }

        // See if we've already computed the same operation.
        let operation = CubicOperation {
            const_0,
            const_1,
            multiplicand_0,
            multiplicand_1,
            addend,
        };
        if let Some(&result) = cubic_results.get(&operation) {
            return result;
        }

        let result = if Self::cubic_target_as_constant(builder, addend) == Some(CubicElement::ZERO)
        {
            // If the addend is zero, we use a multiplication gate.
            Self::compute_mul_cubic_operation(builder, operation)
        } else {
            // Otherwise, we use an arithmetic gate.
            Self::compute_cubic_operation(builder, operation)
        };
        cubic_results.insert(operation, result);
        result
    }

    fn compute_cubic_operation(
        builder: &mut CircuitBuilder<F, D>,
        operation: CubicOperation<F>,
    ) -> CubicElement<Target> {
        let gate = ArithmeticCubicGate::new_from_config(&builder.config);
        let constants = vec![operation.const_0, operation.const_1];
        let (gate, i) = builder.find_slot(gate, &constants, &constants);
        let wires_multiplicand_0 =
            CubicElement::from_range(gate, ArithmeticCubicGate::wires_ith_multiplicand_0(i));
        let wires_multiplicand_1 =
            CubicElement::from_range(gate, ArithmeticCubicGate::wires_ith_multiplicand_1(i));
        let wires_addend = CubicElement::from_range(gate, ArithmeticCubicGate::wires_ith_addend(i));

        Self::connect(builder, operation.multiplicand_0, wires_multiplicand_0);
        Self::connect(builder, operation.multiplicand_1, wires_multiplicand_1);
        Self::connect(builder, operation.addend, wires_addend);

        CubicElement::from_range(gate, ArithmeticCubicGate::wires_ith_output(i))
    }

    fn compute_mul_cubic_operation(
        builder: &mut CircuitBuilder<F, D>,
        operation: CubicOperation<F>,
    ) -> CubicElement<Target> {
        let gate = MulCubicGate::new_from_config(&builder.config);
        let constants = vec![operation.const_0];
        let (gate, i) = builder.find_slot(gate, &constants, &constants);
        let wires_multiplicand_0 =
            CubicElement::from_range(gate, MulCubicGate::wires_ith_multiplicand_0(i));
        let wires_multiplicand_1 =
            CubicElement::from_range(gate, MulCubicGate::wires_ith_multiplicand_1(i));

        Self::connect(builder, operation.multiplicand_0, wires_multiplicand_0);
        Self::connect(builder, operation.multiplicand_1, wires_multiplicand_1);

        CubicElement::from_range(gate, MulCubicGate::wires_ith_output(i))
    }

    fn cubic_extension_special_cases(
        builder: &mut CircuitBuilder<F, D>,
        const_0: F,
        const_1: F,
        multiplicand_0: CubicElement<Target>,
        multiplicand_1: CubicElement<Target>,
        addend: CubicElement<Target>,
    ) -> Option<CubicElement<Target>> {
        let zero = Self::zero(builder);

        let mul_0_const = Self::cubic_target_as_constant(builder, multiplicand_0);
        let mul_1_const = Self::cubic_target_as_constant(builder, multiplicand_1);
        let addend_const = Self::cubic_target_as_constant(builder, addend);

        let first_term_zero =
            const_0 == F::ZERO || multiplicand_0 == zero || multiplicand_1 == zero;
        let second_term_zero = const_1 == F::ZERO || addend == zero;

        // If both terms are constant, return their (constant) sum.
        let first_term_const = if first_term_zero {
            Some(CubicElement::ZERO)
        } else if let (Some(x), Some(y)) = (mul_0_const, mul_1_const) {
            Some(x * y * const_0)
        } else {
            None
        };
        let second_term_const = if second_term_zero {
            Some(CubicElement::ZERO)
        } else {
            addend_const.map(|x| x * const_1)
        };
        if let (Some(x), Some(y)) = (first_term_const, second_term_const) {
            return Some(Self::constant(builder, x + y));
        }

        if first_term_zero && const_1.is_one() {
            return Some(addend);
        }

        if second_term_zero {
            if let Some(x) = mul_0_const {
                if (x * const_0) == CubicElement::ONE {
                    return Some(multiplicand_1);
                }
            }
            if let Some(x) = mul_1_const {
                if (x * const_0) == CubicElement::ONE {
                    return Some(multiplicand_0);
                }
            }
        }

        None
    }

    #[allow(dead_code)]
    pub(crate) fn add(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<Target>,
        b: CubicElement<Target>,
    ) -> CubicElement<Target> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([
            builder.add(x_0, y_0),
            builder.add(x_1, y_1),
            builder.add(x_2, y_2),
        ])
    }

    #[allow(dead_code)]
    pub(crate) fn sub(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<Target>,
        b: CubicElement<Target>,
    ) -> CubicElement<Target> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([
            builder.sub(x_0, y_0),
            builder.sub(x_1, y_1),
            builder.sub(x_2, y_2),
        ])
    }

    #[allow(dead_code)]
    pub(crate) fn scalar_mul(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<Target>,
        scalar: Target,
    ) -> CubicElement<Target> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        CubicElement([
            builder.mul(x_0, scalar),
            builder.mul(x_1, scalar),
            builder.mul(x_2, scalar),
        ])
    }

    #[allow(dead_code)]
    pub(crate) fn mul(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<Target>,
        b: CubicElement<Target>,
    ) -> CubicElement<Target> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);

        let x_0y_0 = builder.mul(x_0, y_0);
        let x_0y_1 = builder.mul(x_0, y_1);
        let x_0y_2 = builder.mul(x_0, y_2);
        let x_1y_0 = builder.mul(x_1, y_0);
        let x_1y_1 = builder.mul(x_1, y_1);
        let x_1y_2 = builder.mul(x_1, y_2);
        let x_2y_0 = builder.mul(x_2, y_0);
        let x_2y_1 = builder.mul(x_2, y_1);
        let x_2y_2 = builder.mul(x_2, y_2);

        let mut z_0 = builder.sub(x_0y_0, x_1y_2);
        z_0 = builder.sub(z_0, x_2y_1);

        let mut z_1 = builder.add(x_0y_1, x_1y_0);
        z_1 = builder.add(z_1, x_1y_2);
        z_1 = builder.add(z_1, x_2y_1);
        z_1 = builder.sub(z_1, x_2y_2);

        let mut z_2 = builder.add(x_0y_2, x_1y_1);
        z_2 = builder.add(z_2, x_2y_0);
        z_2 = builder.add(z_2, x_2y_2);

        CubicElement([z_0, z_1, z_2])
    }

    pub(crate) fn add_extension(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<ExtensionTarget<D>>,
        b: CubicElement<ExtensionTarget<D>>,
    ) -> CubicElement<ExtensionTarget<D>> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([
            builder.add_extension(x_0, y_0),
            builder.add_extension(x_1, y_1),
            builder.add_extension(x_2, y_2),
        ])
    }

    pub(crate) fn sub_extension(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<ExtensionTarget<D>>,
        b: CubicElement<ExtensionTarget<D>>,
    ) -> CubicElement<ExtensionTarget<D>> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([
            builder.sub_extension(x_0, y_0),
            builder.sub_extension(x_1, y_1),
            builder.sub_extension(x_2, y_2),
        ])
    }

    pub(crate) fn scalar_mul_extension(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<ExtensionTarget<D>>,
        scalar: ExtensionTarget<D>,
    ) -> CubicElement<ExtensionTarget<D>> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        CubicElement([
            builder.mul_extension(x_0, scalar),
            builder.mul_extension(x_1, scalar),
            builder.mul_extension(x_2, scalar),
        ])
    }

    pub(crate) fn mul_extension(
        builder: &mut CircuitBuilder<F, D>,
        a: CubicElement<ExtensionTarget<D>>,
        b: CubicElement<ExtensionTarget<D>>,
    ) -> CubicElement<ExtensionTarget<D>> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);

        let x_0y_0 = builder.mul_extension(x_0, y_0);
        let x_0y_1 = builder.mul_extension(x_0, y_1);
        let x_0y_2 = builder.mul_extension(x_0, y_2);
        let x_1y_0 = builder.mul_extension(x_1, y_0);
        let x_1y_1 = builder.mul_extension(x_1, y_1);
        let x_1y_2 = builder.mul_extension(x_1, y_2);
        let x_2y_0 = builder.mul_extension(x_2, y_0);
        let x_2y_1 = builder.mul_extension(x_2, y_1);
        let x_2y_2 = builder.mul_extension(x_2, y_2);

        let mut z_0 = builder.sub_extension(x_0y_0, x_1y_2);
        z_0 = builder.sub_extension(z_0, x_2y_1);

        let mut z_1 = builder.add_extension(x_0y_1, x_1y_0);
        z_1 = builder.add_extension(z_1, x_1y_2);
        z_1 = builder.add_extension(z_1, x_2y_1);
        z_1 = builder.sub_extension(z_1, x_2y_2);

        let mut z_2 = builder.add_extension(x_0y_2, x_1y_1);
        z_2 = builder.add_extension(z_2, x_2y_0);
        z_2 = builder.add_extension(z_2, x_2y_2);

        CubicElement([z_0, z_1, z_2])
    }
}
