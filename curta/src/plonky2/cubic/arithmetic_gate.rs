use core::ops::Range;

use plonky2::field::extension::Extendable;
use plonky2::gates::gate::Gate;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::witness::PartitionWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use super::operations::CubicBuilderOperations;
use crate::math::prelude::cubic::element::CubicElement;

pub(crate) const E: usize = 3;

/// A gate which can perform a weighted multiply-add, i.e. `result = c0 x y + c1 z`. If the config
/// supports enough routed wires, it can support several such operations in one gate.
#[derive(Debug, Clone)]
pub struct ArithmeticCubicGate {
    /// Number of arithmetic operations performed by an arithmetic gate.
    pub num_ops: usize,
}

impl ArithmeticCubicGate {
    pub fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) fn num_ops(config: &CircuitConfig) -> usize {
        let wires_per_op = 4 * E;
        config.num_routed_wires / wires_per_op
    }

    pub fn wires_ith_multiplicand_0(i: usize) -> Range<usize> {
        4 * E * i..4 * E * i + E
    }
    pub fn wires_ith_multiplicand_1(i: usize) -> Range<usize> {
        4 * E * i + E..4 * E * i + 2 * E
    }
    pub fn wires_ith_addend(i: usize) -> Range<usize> {
        4 * E * i + 2 * E..4 * E * i + 3 * E
    }
    pub fn wires_ith_output(i: usize) -> Range<usize> {
        4 * E * i + 3 * E..4 * E * i + 4 * E
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ArithmeticCubicGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self { num_ops })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops * E);
        let get_cubic =
            |range: Range<usize>| CubicElement(vars.local_wires[range].try_into().unwrap());
        for i in 0..self.num_ops {
            let multiplicand_0 = get_cubic(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = get_cubic(Self::wires_ith_multiplicand_1(i));
            let addend = get_cubic(Self::wires_ith_addend(i));
            let output = get_cubic(Self::wires_ith_output(i));
            let computed_output = (multiplicand_0 * multiplicand_1) * const_0 + addend * const_1;

            constraints.extend((output - computed_output).as_array());
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let get_cubic = |range: Range<usize>| {
            CubicElement(
                range
                    .map(|i| vars.local_wires[i])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
        };

        for i in 0..self.num_ops {
            let multiplicand_0 = get_cubic(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = get_cubic(Self::wires_ith_multiplicand_1(i));
            let addend = get_cubic(Self::wires_ith_addend(i));
            let output = get_cubic(Self::wires_ith_output(i));
            let computed_output = (multiplicand_0 * multiplicand_1) * const_0 + addend * const_1;

            yield_constr.many((output - computed_output).as_array());
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops * E);
        let get_cubic =
            |range: Range<usize>| CubicElement(vars.local_wires[range].try_into().unwrap());
        for i in 0..self.num_ops {
            let multiplicand_0 = get_cubic(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = get_cubic(Self::wires_ith_multiplicand_1(i));
            let addend = get_cubic(Self::wires_ith_addend(i));
            let output = get_cubic(Self::wires_ith_output(i));
            let computed_output = {
                let mul =
                    CubicBuilderOperations::mul_extension(builder, multiplicand_0, multiplicand_1);
                let scaled_mul =
                    CubicBuilderOperations::scalar_mul_extension(builder, mul, const_0);
                let addend_mul =
                    CubicBuilderOperations::scalar_mul_extension(builder, addend, const_1);
                CubicBuilderOperations::add_extension(builder, scaled_mul, addend_mul)
            };

            let diff = CubicBuilderOperations::sub_extension(builder, output, computed_output);
            constraints.extend(diff.0);
        }

        constraints
    }

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    ArithmeticCubicGenerator {
                        row,
                        const_0: local_constants[0],
                        const_1: local_constants[1],
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * 4 * E
    }

    fn num_constants(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * E
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArithmeticCubicGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    const_0: F,
    const_1: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticCubicGenerator<F, D> {
    pub fn id() -> String {
        "ArithmeticCubicGenerator".to_string()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for ArithmeticCubicGenerator<F, D>
{
    fn id(&self) -> String {
        Self::id()
    }

    fn dependencies(&self) -> Vec<Target> {
        ArithmeticCubicGate::wires_ith_multiplicand_0(self.i)
            .chain(ArithmeticCubicGate::wires_ith_multiplicand_1(self.i))
            .chain(ArithmeticCubicGate::wires_ith_addend(self.i))
            .map(|i| Target::wire(self.row, i))
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let extract_extension = |range: Range<usize>| -> CubicElement<F> {
            let t = CubicElement::from_range(self.row, range);
            t.get(witness)
        };

        let multiplicand_0 =
            extract_extension(ArithmeticCubicGate::wires_ith_multiplicand_0(self.i));
        let multiplicand_1 =
            extract_extension(ArithmeticCubicGate::wires_ith_multiplicand_1(self.i));
        let addend = extract_extension(ArithmeticCubicGate::wires_ith_addend(self.i));

        let output_target =
            CubicElement::from_range(self.row, ArithmeticCubicGate::wires_ith_output(self.i));

        let computed_output =
            (multiplicand_0 * multiplicand_1) * self.const_0 + addend * self.const_1;

        output_target.set(&computed_output, out_buffer);
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)?;
        dst.write_field(self.const_1)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        let const_1 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            row,
            const_0,
            const_1,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::ArithmeticCubicGate;

    #[test]
    fn low_degree() {
        let gate =
            ArithmeticCubicGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate =
            ArithmeticCubicGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }
}
