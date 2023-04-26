//! Arithmetic expressions
//!
//! This module defines arithmetic expressions that can be used to define constraint
//! equations in column entries. The arithmetic expressions are defined in terms of the
//! `ArithmeticExpressionSlice` type.
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
pub struct ArithmeticExpression<F, const D: usize> {
    pub(crate) expression: ArithmeticExpressionSlice<F, D>,
    pub(crate) size: usize,
}

impl<F, const D: usize> ArithmeticExpression<F, D> {
    pub fn from_constant_vec(constants: Vec<F>) -> Self {
        let size = constants.len();
        Self {
            expression: ArithmeticExpressionSlice::Const(constants),
            size,
        }
    }

    pub fn from_constant(constant: F) -> Self {
        Self::from_constant_vec(vec![constant])
    }
}

#[derive(Clone, Debug)]
pub enum ArithmeticExpressionSlice<F, const D: usize> {
    /// A contiguous chunk of elemnt of a trace column.
    Input(MemorySlice),
    /// A constant vector of field values.
    Const(Vec<F>),
    /// The addition of two arithmetic expressions.
    Add(
        Arc<ArithmeticExpressionSlice<F, D>>,
        Arc<ArithmeticExpressionSlice<F, D>>,
    ),
    /// The subtraction of two arithmetic expressions
    Sub(
        Arc<ArithmeticExpressionSlice<F, D>>,
        Arc<ArithmeticExpressionSlice<F, D>>,
    ),
    /// The scalar multiplication of an arithmetic expression by a field element.
    ScalarMul(F, Arc<ArithmeticExpressionSlice<F, D>>),
    /// The multiplication of two arithmetic expressions.
    Mul(
        Arc<ArithmeticExpressionSlice<F, D>>,
        Arc<ArithmeticExpressionSlice<F, D>>,
    ),
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticExpressionSlice<F, D> {
    pub fn new<T: Register>(input: &T) -> Self {
        ArithmeticExpressionSlice::Input(*input.register())
    }

    pub fn from_raw_register(input: MemorySlice) -> Self {
        ArithmeticExpressionSlice::Input(input)
    }

    pub fn from_constant(constant: F) -> Self {
        ArithmeticExpressionSlice::Const(vec![constant])
    }

    pub fn from_constant_vec(constants: Vec<F>) -> Self {
        ArithmeticExpressionSlice::Const(constants)
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
            ArithmeticExpressionSlice::Input(input) => input.packed_generic_vars(vars).to_vec(),
            ArithmeticExpressionSlice::Const(constants) => {
                let s = |x: &F| P::from(FE::from_basefield(*x));
                constants.iter().map(s).collect()
            }
            ArithmeticExpressionSlice::Add(left, right) => {
                let left = left.packed_generic(vars);
                let right = right.packed_generic(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r + *l)
                    .collect()
            }
            ArithmeticExpressionSlice::Sub(left, right) => {
                let left = left.packed_generic(vars);
                let right = right.packed_generic(vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| *r - *l)
                    .collect()
            }
            ArithmeticExpressionSlice::ScalarMul(scalar, expr) => {
                let expr = expr.packed_generic(vars);
                let s = FE::from_basefield(*scalar);
                expr.iter().map(|e| *e * s).collect()
            }
            ArithmeticExpressionSlice::Mul(left, right) => {
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
            ArithmeticExpressionSlice::Input(input) => input.ext_circuit_vars(vars).to_vec(),
            ArithmeticExpressionSlice::Const(constants) => constants
                .iter()
                .map(|x| builder.constant_extension(F::Extension::from_basefield(*x)))
                .collect(),
            ArithmeticExpressionSlice::Add(left, right) => {
                let left = left.ext_circuit(builder, vars);
                let right = right.ext_circuit(builder, vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| builder.add_extension(*l, *r))
                    .collect()
            }
            ArithmeticExpressionSlice::Sub(left, right) => {
                let left = left.ext_circuit(builder, vars);
                let right = right.ext_circuit(builder, vars);
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| builder.sub_extension(*l, *r))
                    .collect()
            }
            ArithmeticExpressionSlice::ScalarMul(scalar, expr) => {
                let expr = expr.ext_circuit(builder, vars);
                expr.iter()
                    .map(|x| builder.mul_const_extension(*scalar, *x))
                    .collect()
            }
            ArithmeticExpressionSlice::Mul(left, right) => {
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
        assert_eq!(
            self.size, rhs.size,
            "Cannot add arithmetic expressions of different sizes"
        );
        Self {
            expression: ArithmeticExpressionSlice::Add(
                Arc::new(self.expression),
                Arc::new(rhs.expression),
            ),
            size: self.size,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Sub for ArithmeticExpression<F, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.size, rhs.size,
            "Cannot subtract arithmetic expressions of different sizes"
        );
        Self {
            expression: ArithmeticExpressionSlice::Sub(
                Arc::new(self.expression),
                Arc::new(rhs.expression),
            ),
            size: self.size,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Add<Vec<F>> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn add(self, rhs: Vec<F>) -> Self::Output {
        assert_eq!(
            self.size,
            rhs.len(),
            "Cannot add vector of size {} arithmetic expression of size {}",
            rhs.len(),
            self.size
        );
        Self {
            expression: ArithmeticExpressionSlice::Add(
                Arc::new(self.expression),
                Arc::new(ArithmeticExpressionSlice::Const(rhs)),
            ),
            size: self.size,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Sub<Vec<F>> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn sub(self, rhs: Vec<F>) -> Self::Output {
        assert_eq!(
            self.size,
            rhs.len(),
            "Cannot subtract a vector of size {} arithmetic expression of size {}",
            rhs.len(),
            self.size
        );
        Self {
            expression: ArithmeticExpressionSlice::Sub(
                Arc::new(self.expression),
                Arc::new(ArithmeticExpressionSlice::Const(rhs)),
            ),
            size: self.size,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Add<F> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        self + vec![rhs]
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Sub<F> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        self - vec![rhs]
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Mul<F> for ArithmeticExpression<F, D> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self {
            expression: ArithmeticExpressionSlice::ScalarMul(rhs, Arc::new(self.expression)),
            size: self.size,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Mul for ArithmeticExpression<F, D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.size, rhs.size) {
            (1, _) => Self {
                expression: ArithmeticExpressionSlice::Mul(
                    Arc::new(self.expression),
                    Arc::new(rhs.expression),
                ),
                size: rhs.size,
            },
            (_, 1) => Self {
                expression: ArithmeticExpressionSlice::Mul(
                    Arc::new(self.expression),
                    Arc::new(rhs.expression),
                ),
                size: self.size,
            },
            (n, m) if n == m => Self {
                expression: ArithmeticExpressionSlice::Mul(
                    Arc::new(self.expression),
                    Arc::new(rhs.expression),
                ),
                size: n,
            },
            _ => panic!("Cannot multiply arithmetic expressions of different sizes"),
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Sample;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::arithmetic::builder::StarkBuilder;
    use crate::arithmetic::chip::{StarkParameters, TestStark};
    use crate::arithmetic::instruction::write::WriteInstruction;
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

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D>
        for TestArithmeticExpression<F, D>
    {
        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 3;

        type Instruction = WriteInstruction;
    }

    #[derive(Clone, Debug)]
    pub struct Test2ArithmeticExpression<F, const D: usize> {
        _marker: core::marker::PhantomData<F>,
    }

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D>
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

        let mut builder = StarkBuilder::<Test2ArithmeticExpression<F, D>, F, D>::new();

        let input_1 = builder.alloc::<U16Register>();
        let input_2 = builder.alloc::<U16Register>();
        let output = builder.alloc::<U16Register>();

        builder.write_data(&input_1).unwrap();
        builder.write_data(&input_2).unwrap();
        builder.write_data(&output).unwrap();

        builder.mul(&input_1, &input_2, &output);

        let (chip, spec) = builder.build();

        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        for i in 0..num_rows {
            let a_val = F::ONES + F::ONES;
            let b_val = F::ONES + F::ONES + F::ONES;
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
