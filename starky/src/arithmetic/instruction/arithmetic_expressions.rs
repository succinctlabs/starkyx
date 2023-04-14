//! Arithmetic expressions
//!
//! This module defines arithmetic expressions that can be used to define constraint
//! equations in column entries. The arithmetic expressions are defined in terms of the
//! `ArithmeticExpression` type.
//!

use core::ops::{Add, Mul, Sub};
use std::sync::Arc;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::arithmetic::register::{MemorySlice, Register};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

/// An arithmetic expression of trace column entries.
///
/// Arithmetic expressions represent a composition of additions, multiplications, subtractions,
/// and scalar multiplications of trace column entries. These expression can be used in
/// an EqualityConstraint to define a constraint equation.
#[derive(Clone, Debug)]
pub enum ArithmeticExpression<F, const D: usize> {
    /// A contiguous chunk of elemnt of a trace column.
    Input(MemorySlice),
    /// A constant vector of field values.
    Const(Vec<F>),
    /// The addition of two arithmetic expressions.
    Add(
        Arc<ArithmeticExpression<F, D>>,
        Arc<ArithmeticExpression<F, D>>,
    ),
    /// The subtraction of two arithmetic expressions.
    Sub(
        Arc<ArithmeticExpression<F, D>>,
        Arc<ArithmeticExpression<F, D>>,
    ),
    /// The scalar multiplication of an arithmetic expression by a field element.
    ScalarMul(F, Arc<ArithmeticExpression<F, D>>),
    /// The multiplication of two arithmetic expressions.
    Mul(
        Arc<ArithmeticExpression<F, D>>,
        Arc<ArithmeticExpression<F, D>>,
    ),
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticExpression<F, D> {
    pub fn new<T: Register>(input: &T) -> Self {
        ArithmeticExpression::Input(*input.register())
    }

    pub fn from_raw_register(input: MemorySlice) -> Self {
        ArithmeticExpression::Input(input)
    }

    pub fn from_constant(constant: F) -> Self {
        ArithmeticExpression::Const(vec![constant])
    }

    pub fn from_constant_vec(constants: Vec<F>) -> Self {
        ArithmeticExpression::Const(constants)
    }

    pub fn packed_generic<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: &StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<P>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            ArithmeticExpression::Input(input) => input.packed_entries_slice(vars).to_vec(),
            ArithmeticExpression::Const(constants) => {
                let s = |x: &F| P::from(FE::from_basefield(*x));
                constants.iter().map(s).collect()
            }
            ArithmeticExpression::Add(left, right) => {
                let left = left.packed_generic(vars);
                let right = right.packed_generic(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r + *l)
                    .collect()
            }
            ArithmeticExpression::Sub(left, right) => {
                let left = left.packed_generic(vars);
                let right = right.packed_generic(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r - *l)
                    .collect()
            }
            ArithmeticExpression::ScalarMul(scalar, expr) => {
                let expr = expr.packed_generic(vars);
                let s = FE::from_basefield(*scalar);
                expr.iter().map(|e| *e * s).collect()
            }
            ArithmeticExpression::Mul(left, right) => {
                let left = left.packed_generic(vars);
                let right = right.packed_generic(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *l * *r)
                    .collect()
            }
        }
    }

    pub fn ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<ExtensionTarget<D>> {
        match self {
            ArithmeticExpression::Input(input) => input.evaluation_targets(vars).to_vec(),
            ArithmeticExpression::Const(constants) => constants
                .iter()
                .map(|x| builder.constant_extension(F::Extension::from_basefield(*x)))
                .collect(),
            ArithmeticExpression::Add(left, right) => {
                let left = left.ext_circuit(builder, vars);
                let right = right.ext_circuit(builder, vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| builder.add_extension(*l, *r))
                    .collect()
            }
            ArithmeticExpression::Sub(left, right) => {
                let left = left.ext_circuit(builder, vars);
                let right = right.ext_circuit(builder, vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| builder.sub_extension(*l, *r))
                    .collect()
            }
            ArithmeticExpression::ScalarMul(scalar, expr) => {
                let expr = expr.ext_circuit(builder, vars);
                expr.iter()
                    .map(|x| builder.mul_const_extension(*scalar, *x))
                    .collect()
            }
            ArithmeticExpression::Mul(left, right) => {
                let left_vals = left.ext_circuit(builder, vars);
                let right_vals = right.ext_circuit(builder, vars);
                left_vals
                    .iter()
                    .zip(right_vals.iter())
                    .map(|(l, r)| builder.mul_extension(*l, *r))
                    .collect()
            }
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Add for ArithmeticExpression<F, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ArithmeticExpression::Add(Arc::new(self), Arc::new(rhs))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Sub for ArithmeticExpression<F, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        ArithmeticExpression::Sub(Arc::new(self), Arc::new(rhs))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Add<Vec<F>> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn add(self, rhs: Vec<F>) -> Self::Output {
        ArithmeticExpression::Add(Arc::new(self), Arc::new(ArithmeticExpression::Const(rhs)))
    }
}

<<<<<<< HEAD
impl<F: RichField + Extendable<D>, const D: usize> Sub<Vec<F>> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn sub(self, rhs: Vec<F>) -> Self::Output {
        ArithmeticExpression::Sub(Arc::new(self), Arc::new(ArithmeticExpression::Const(rhs)))
    }
}

=======
>>>>>>> main
impl<F: RichField + Extendable<D>, const D: usize> Add<F> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        self + vec![rhs]
    }
}

<<<<<<< HEAD
impl<F: RichField + Extendable<D>, const D: usize> Sub<F> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        self - vec![rhs]
    }
}

=======
>>>>>>> main
impl<F: RichField + Extendable<D>, const D: usize> Mul<F> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        ArithmeticExpression::ScalarMul(rhs, Arc::new(self))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Mul for ArithmeticExpression<F, D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        ArithmeticExpression::Mul(Arc::new(self), Arc::new(rhs))
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::{Field, Sample};
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::arithmetic::builder::ChipBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
    use crate::arithmetic::instruction::write::WriteInstruction;
    use crate::arithmetic::instruction::EqualityConstraint;
    use crate::arithmetic::register::U16Register;
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug)]
    pub struct TestArithmeticExpression<F, const D: usize> {
        _marker: core::marker::PhantomData<F>,
    }

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D>
        for TestArithmeticExpression<F, D>
    {
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 3;

        type Instruction = WriteInstruction;
    }

    #[test]
    fn test_arithmetic_expression() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = TestStark<TestArithmeticExpression<F, D>, F, D>;

        let mut builder = ChipBuilder::<TestArithmeticExpression<F, D>, F, D>::new();

        let input_1 = builder.get_local_memory(1).unwrap();
        let input_2 = builder.get_local_memory(1).unwrap();
        let output = builder.get_local_memory(1).unwrap();

        builder.write_raw_register(&input_1).unwrap();
        builder.write_raw_register(&input_2).unwrap();
        builder.write_raw_register(&output).unwrap();

        let in_1 = ArithmeticExpression::<F, D>::Input(input_1);
        let in_2 = ArithmeticExpression::<F, D>::Input(input_2);
        let two = F::ONE + F::ONE;
        let expr = in_1.clone() * in_2 + in_1 * two;

        let out_expr = ArithmeticExpression::<F, D>::Input(output);

        let equal_consr = EqualityConstraint::ArithmeticConstraint(out_expr, expr);

        builder.insert_raw_constraint(equal_consr);

        let (chip, spec) = builder.build();

        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        for i in 0..num_rows {
            let a_val = F::rand();
            let b_val = F::rand();
            let c_val = a_val * b_val + a_val * two;
            handle.write_unsafe_raw(i, &input_1, vec![a_val]).unwrap();
            handle.write_unsafe_raw(i, &input_2, vec![b_val]).unwrap();
            handle.write_unsafe_raw(i, &output, vec![c_val]).unwrap();
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows).unwrap();

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);

        // Verify proof as a stark
        let proof = prove::<F, C, S, D>(
            stark.clone(),
            &config,
            trace,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof = add_virtual_stark_proof_with_pis(
            &mut recursive_builder,
            stark.clone(),
            &config,
            degree_bits,
        );

        recursive_builder.print_gate_counts(0);

        let mut rec_pw = PartialWitness::new();
        set_stark_proof_with_pis_target(&mut rec_pw, &virtual_proof, &proof);

        verify_stark_proof_circuit::<F, C, S, D>(
            &mut recursive_builder,
            stark,
            virtual_proof,
            &config,
        );

        let recursive_data = recursive_builder.build::<C>();

        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut timing,
        )
        .unwrap();

        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }

    #[derive(Clone, Debug)]
    pub struct Test2ArithmeticExpression<F, const D: usize> {
        _marker: core::marker::PhantomData<F>,
    }

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D>
        for Test2ArithmeticExpression<F, D>
    {
        const NUM_ARITHMETIC_COLUMNS: usize = 3;
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = WriteInstruction;
    }

    #[test]
    fn test_register_arithmetic_expression() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = TestStark<Test2ArithmeticExpression<F, D>, F, D>;

        let mut builder = ChipBuilder::<Test2ArithmeticExpression<F, D>, F, D>::new();

        let input_1 = builder.alloc_local::<U16Register>().unwrap();
        let input_2 = builder.alloc_local::<U16Register>().unwrap();
        let output = builder.alloc_local::<U16Register>().unwrap();

        builder.write_data(&input_1).unwrap();
        builder.write_data(&input_2).unwrap();
        builder.write_data(&output).unwrap();

        builder.mul(&input_1, &input_2, &output);

        let (chip, spec) = builder.build();

        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        for i in 0..num_rows {
            let a_val = F::ONE;
            let b_val = F::ONE + F::ONE + F::ONE;
            let c_val = a_val * b_val;
            handle.write_data(i, input_1, vec![a_val]).unwrap();
            handle.write_data(i, input_2, vec![b_val]).unwrap();
            handle.write_data(i, output, vec![c_val]).unwrap();
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows).unwrap();

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);

        // Verify proof as a stark
        let proof = prove::<F, C, S, D>(
            stark.clone(),
            &config,
            trace,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof = add_virtual_stark_proof_with_pis(
            &mut recursive_builder,
            stark.clone(),
            &config,
            degree_bits,
        );

        recursive_builder.print_gate_counts(0);

        let mut rec_pw = PartialWitness::new();
        set_stark_proof_with_pis_target(&mut rec_pw, &virtual_proof, &proof);

        verify_stark_proof_circuit::<F, C, S, D>(
            &mut recursive_builder,
            stark,
            virtual_proof,
            &config,
        );

        let recursive_data = recursive_builder.build::<C>();

        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut timing,
        )
        .unwrap();

        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }
}
