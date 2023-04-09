use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::field::{FieldParameters, FieldRegister};
use super::*;
use crate::arithmetic::instruction::Instruction;
use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::arithmetic::register::{DataRegister, WitnessData};
use crate::arithmetic::util::{extract_witness_and_shift, split_digits, to_field_iter};
use crate::arithmetic::{ArithmeticParser, Register};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub struct FpMull<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> {
    a: FieldRegister<P, N_LIMBS>,
    b: FieldRegister<P, N_LIMBS>,
    result: FieldRegister<P, N_LIMBS>,
    carry: Option<Register>,
    witness_low: Option<Register>,
    witness_high: Option<Register>,
}

impl<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> FpMull<P, N_LIMBS> {
    const NUM_CARRY_LIMBS: usize = N_LIMBS;
    pub const NUM_WITNESS_LOW_LIMBS: usize = 2 * N_LIMBS - 2;
    pub const NUM_WITNESS_HIGH_LIMBS: usize = 2 * N_LIMBS - 2;

    #[inline]
    pub const fn new(
        a: FieldRegister<P, N_LIMBS>,
        b: FieldRegister<P, N_LIMBS>,
        result: FieldRegister<P, N_LIMBS>,
    ) -> Self {
        // let (carry, witness_low, witness_high) = match witness {
        //     Register::Local(index, _) => (
        //         Register::Local(index, NUM_CARRY_LIMBS),
        //         Register::Local(index + NUM_CARRY_LIMBS, NUM_WITNESS_LIMBS),
        //         Register::Local(
        //             index + NUM_CARRY_LIMBS + NUM_WITNESS_LIMBS,
        //             NUM_WITNESS_LIMBS,
        //         ),
        //     ),
        //     Register::Next(index, _) => (
        //         Register::Next(index, NUM_CARRY_LIMBS),
        //         Register::Next(index + NUM_CARRY_LIMBS, NUM_WITNESS_LIMBS),
        //         Register::Next(
        //             index + NUM_CARRY_LIMBS + NUM_WITNESS_LIMBS,
        //             NUM_WITNESS_LIMBS,
        //         ),
        //     ),
        // };
        Self {
            a,
            b,
            result,
            carry: None,
            witness_low: None,
            witness_high: None,
        }
    }

    #[inline]
    pub fn assign_row<T: Copy>(&self, trace_rows: &mut [Vec<T>], row: &mut [T], row_index: usize) {
        let mut index = 0;
        self.result
            .register()
            .assign(trace_rows, &mut row[index..N_LIMBS], row_index);
        index += N_LIMBS;
        self.carry.unwrap().assign(
            trace_rows,
            &mut row[index..index + Self::NUM_CARRY_LIMBS],
            row_index,
        );
        index += Self::NUM_CARRY_LIMBS;
        self.witness_low.unwrap().assign(
            trace_rows,
            &mut row[index..index + Self::NUM_WITNESS_LOW_LIMBS],
            row_index,
        );
        index += Self::NUM_WITNESS_LOW_LIMBS;
        self.witness_high.unwrap().assign(
            trace_rows,
            &mut row[index..index + Self::NUM_WITNESS_HIGH_LIMBS],
            row_index,
        );
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const N: usize, FP: FieldParameters<N>>
    Instruction<F, D> for FpMull<FP, N>
{
    fn shift_right(&mut self, free_shift: usize, arithmetic_shift: usize) {
        self.a.shift_right(free_shift, arithmetic_shift);
        self.b.shift_right(free_shift, arithmetic_shift);
        self.result.shift_right(free_shift, arithmetic_shift);
        if let Some(mut c) = self.carry {
            c.shift_right(arithmetic_shift);
        }
        if let Some(mut w) = self.witness_low {
            w.shift_right(arithmetic_shift);
        }
        if let Some(mut w) = self.witness_high {
            w.shift_right(arithmetic_shift);
        }
    }

    fn memory_vec(&self) -> Vec<Register> {
        vec![
            *self.a.register(),
            *self.b.register(),
            *self.result.register(),
        ]
    }

    fn witness_data(&self) -> Option<WitnessData> {
        Some(WitnessData::u16(
            Self::NUM_CARRY_LIMBS + Self::NUM_WITNESS_LOW_LIMBS + Self::NUM_WITNESS_HIGH_LIMBS,
        ))
    }

    fn set_witness(&mut self, witness: Register) -> Result<()> {
        let (carry, witness_low, witness_high) = match witness {
            Register::Local(index, _) => (
                Register::Local(index, Self::NUM_CARRY_LIMBS),
                Register::Local(index + Self::NUM_CARRY_LIMBS, Self::NUM_WITNESS_LOW_LIMBS),
                Register::Local(
                    index + Self::NUM_CARRY_LIMBS + Self::NUM_WITNESS_LOW_LIMBS,
                    Self::NUM_WITNESS_HIGH_LIMBS,
                ),
            ),
            Register::Next(index, _) => (
                Register::Next(index, Self::NUM_CARRY_LIMBS),
                Register::Next(index + Self::NUM_CARRY_LIMBS, Self::NUM_WITNESS_LOW_LIMBS),
                Register::Next(
                    index + Self::NUM_CARRY_LIMBS + Self::NUM_WITNESS_LOW_LIMBS,
                    Self::NUM_WITNESS_HIGH_LIMBS,
                ),
            ),
        };
        self.carry = Some(carry);
        self.witness_low = Some(witness_low);
        self.witness_high = Some(witness_high);

        Ok(())
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.assign_row(trace_rows, row, row_index)
    }

    fn packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // get all the data
        let a = self.a.register().packed_entries_slice(&vars);
        let b = self.b.register().packed_entries_slice(&vars);
        let result = self.result.register().packed_entries_slice(&vars);

        let carry = self.carry.unwrap().packed_entries_slice(&vars);
        let witness_low = self.witness_low.unwrap().packed_entries_slice(&vars);
        let witness_high = self.witness_high.unwrap().packed_entries_slice(&vars);

        // Construct the expected vanishing polynmial
        let ab = PolynomialOps::mul(a, b);
        let ab_minus_result = PolynomialOps::sub(&ab, result);
        let p_limbs = Polynomial::<FE>::from_iter(P_iter());
        let mul_times_carry = PolynomialOps::scalar_poly_mul(carry, p_limbs.as_slice());
        let vanishing_poly = PolynomialOps::sub(&ab_minus_result, &mul_times_carry);

        // reconstruct witness

        let limb = FE::from_canonical_u32(LIMB);

        // Reconstruct and shift back the witness polynomial
        let w_shifted = witness_low
            .iter()
            .zip(witness_high.iter())
            .map(|(x, y)| *x + (*y * limb));

        let offset = FE::from_canonical_u32(WITNESS_OFFSET as u32);
        let w = w_shifted.map(|x| x - offset).collect::<Vec<P>>();

        // Multiply by (x-2^16) and make the constraint
        let root_monomial: &[P] = &[P::from(-limb), P::from(P::Scalar::ONE)];
        let witness_times_root = PolynomialOps::mul(&w, root_monomial);

        //debug_assert!(vanishing_poly.len() == witness_times_root.len());
        for i in 0..vanishing_poly.len() {
            yield_constr.constraint_transition(vanishing_poly[i] - witness_times_root[i]);
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // get all the data
        let a = self.a.register().evaluation_targets(&vars);
        let b = self.b.register().evaluation_targets(&vars);
        let result = self.result.register().evaluation_targets(&vars);

        let carry = self.carry.unwrap().evaluation_targets(&vars);
        let witness_low = self.witness_low.unwrap().evaluation_targets(&vars);
        let witness_high = self.witness_high.unwrap().evaluation_targets(&vars);

        // Construct the expected vanishing polynmial
        let ab = PolynomialGadget::mul_extension(builder, a, b);
        let ab_minus_result = PolynomialGadget::sub_extension(builder, &ab, result);
        let p_limbs =
            PolynomialGadget::constant_extension(builder, &P_iter().collect::<Vec<_>>()[..]);
        let mul_times_carry = PolynomialGadget::mul_extension(builder, carry, &p_limbs[..]);
        let vanishing_poly =
            PolynomialGadget::sub_extension(builder, &ab_minus_result, &mul_times_carry);

        // reconstruct witness

        // Reconstruct and shift back the witness polynomial
        let limb_const = F::Extension::from_canonical_u32(2u32.pow(16));
        let limb = builder.constant_extension(limb_const);
        let w_high_times_limb =
            PolynomialGadget::ext_scalar_mul_extension(builder, witness_high, &limb);
        let w_shifted = PolynomialGadget::add_extension(builder, witness_low, &w_high_times_limb);
        let offset =
            builder.constant_extension(F::Extension::from_canonical_u32(WITNESS_OFFSET as u32));
        let w = PolynomialGadget::sub_constant_extension(builder, &w_shifted, &offset);

        // Multiply by (x-2^16) and make the constraint
        let neg_limb = builder.constant_extension(-limb_const);
        let root_monomial = &[neg_limb, builder.constant_extension(F::Extension::ONE)];
        let witness_times_root =
            PolynomialGadget::mul_extension(builder, w.as_slice(), root_monomial);

        let constraint =
            PolynomialGadget::sub_extension(builder, &vanishing_poly, &witness_times_root);
        for constr in constraint {
            yield_constr.constraint_transition(builder, constr);
        }
    }
}

/// Trace row for fp_mul operation
///
/// Returns a vector
/// [Input[2 * N_LIMBS], output[N_LIMBS], carry[NUM_CARRY_LIMBS], Witness_low[NUM_WITNESS_LIMBS], Witness_high[NUM_WITNESS_LIMBS]]
pub fn fpmul_trace<F: RichField + Extendable<D>, const D: usize>(
    a: BigUint,
    b: BigUint,
) -> (Vec<F>, BigUint) {
    let p = get_p();
    let result = (&a * &b) % &p;
    debug_assert!(result < p);
    let carry = (&a * &b - &result) / &p;
    debug_assert!(carry < p);
    debug_assert_eq!(&carry * &p, &a * &b - &result);

    // make polynomial limbs
    let p_a = Polynomial::<i64>::from_biguint_num(&a, 16, N_LIMBS);
    let p_b = Polynomial::<i64>::from_biguint_num(&b, 16, N_LIMBS);
    let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, N_LIMBS);

    let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
    let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, NUM_CARRY_LIMBS);

    // Compute the vanishing polynomial
    let vanishing_poly = &p_a * &p_b - &p_result - &p_carry * &p_p;
    debug_assert_eq!(vanishing_poly.degree(), NUM_WITNESS_LIMBS);

    // Compute the witness
    let witness_shifted = extract_witness_and_shift(&vanishing_poly, WITNESS_OFFSET as u32);
    let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

    let mut row = Vec::with_capacity(NUM_MUL_COLUMNS);

    // output
    row.extend(to_field_iter::<F>(&p_result));
    // carry and witness
    row.extend(to_field_iter::<F>(&p_carry));
    row.extend(witness_low);
    row.extend(witness_high);

    (row, result)
}

//////////////////////////////////////// OLD CODE /////////////////////////////////////

pub const NUM_CARRY_LIMBS: usize = N_LIMBS;
pub const NUM_WITNESS_LIMBS: usize = 2 * N_LIMBS - 2;
const NUM_MUL_COLUMNS: usize = 3 * N_LIMBS + NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS;
pub const TOTAL_WITNESS_COLUMNS: usize = NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS;

/// A gadget to compute
/// QUAD(x, y, z, w) = (a * b + c * d) mod p
#[derive(Debug, Clone, Copy)]
pub struct FpMulLayout {
    a: Register,
    b: Register,
    output: Register,
    witness: Register,
    carry: Register,
    witness_low: Register,
    witness_high: Register,
}

impl FpMulLayout {
    #[inline]
    pub const fn new(a: Register, b: Register, output: Register, witness: Register) -> Self {
        let (carry, witness_low, witness_high) = match witness {
            Register::Local(index, _) => (
                Register::Local(index, NUM_CARRY_LIMBS),
                Register::Local(index + NUM_CARRY_LIMBS, NUM_WITNESS_LIMBS),
                Register::Local(
                    index + NUM_CARRY_LIMBS + NUM_WITNESS_LIMBS,
                    NUM_WITNESS_LIMBS,
                ),
            ),
            Register::Next(index, _) => (
                Register::Next(index, NUM_CARRY_LIMBS),
                Register::Next(index + NUM_CARRY_LIMBS, NUM_WITNESS_LIMBS),
                Register::Next(
                    index + NUM_CARRY_LIMBS + NUM_WITNESS_LIMBS,
                    NUM_WITNESS_LIMBS,
                ),
            ),
        };
        Self {
            a,
            b,
            output,
            witness,
            carry,
            witness_low,
            witness_high,
        }
    }

    #[inline]
    pub fn assign_row<T: Copy>(&self, trace_rows: &mut [Vec<T>], row: &mut [T], row_index: usize) {
        self.output
            .assign(trace_rows, &mut row[0..N_LIMBS], row_index);
        self.witness.assign(
            trace_rows,
            &mut row[N_LIMBS..N_LIMBS + NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS],
            row_index,
        )
    }
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    /// Returns a vector
    /// [Input[2 * N_LIMBS], output[N_LIMBS], carry[NUM_CARRY_LIMBS], Witness_low[NUM_WITNESS_LIMBS], Witness_high[NUM_WITNESS_LIMBS]]
    pub fn fpmul_trace(a: BigUint, b: BigUint) -> (Vec<F>, BigUint) {
        let p = get_p();
        let result = (&a * &b) % &p;
        debug_assert!(result < p);
        let carry = (&a * &b - &result) / &p;
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, &a * &b - &result);

        // make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(&a, 16, N_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(&b, 16, N_LIMBS);
        let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, N_LIMBS);

        let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
        let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, NUM_CARRY_LIMBS);

        // Compute the vanishing polynomial
        let vanishing_poly = &p_a * &p_b - &p_result - &p_carry * &p_p;
        debug_assert_eq!(vanishing_poly.degree(), NUM_WITNESS_LIMBS);

        // Compute the witness
        let witness_shifted = extract_witness_and_shift(&vanishing_poly, WITNESS_OFFSET as u32);
        let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

        let mut row = Vec::with_capacity(NUM_MUL_COLUMNS);

        // output
        row.extend(to_field_iter::<F>(&p_result));
        // carry and witness
        row.extend(to_field_iter::<F>(&p_carry));
        row.extend(witness_low);
        row.extend(witness_high);

        (row, result)
    }

    /// Quad generic constraints
    pub fn fpmul_packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        layout: FpMulLayout,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // get all the data
        let a = layout.a.packed_entries_slice(&vars);
        let b = layout.b.packed_entries_slice(&vars);
        let output = layout.output.packed_entries_slice(&vars);

        let carry = layout.carry.packed_entries_slice(&vars);
        let witness_low = layout.witness_low.packed_entries_slice(&vars);
        let witness_high = layout.witness_high.packed_entries_slice(&vars);

        // Construct the expected vanishing polynmial
        let ab = PolynomialOps::mul(a, b);
        let ab_minus_output = PolynomialOps::sub(&ab, output);
        let p_limbs = Polynomial::<FE>::from_iter(P_iter());
        let mul_times_carry = PolynomialOps::scalar_poly_mul(carry, p_limbs.as_slice());
        let vanishing_poly = PolynomialOps::sub(&ab_minus_output, &mul_times_carry);

        // reconstruct witness

        let limb = FE::from_canonical_u32(LIMB);

        // Reconstruct and shift back the witness polynomial
        let w_shifted = witness_low
            .iter()
            .zip(witness_high.iter())
            .map(|(x, y)| *x + (*y * limb));

        let offset = FE::from_canonical_u32(WITNESS_OFFSET as u32);
        let w = w_shifted.map(|x| x - offset).collect::<Vec<P>>();

        // Multiply by (x-2^16) and make the constraint
        let root_monomial: &[P] = &[P::from(-limb), P::from(P::Scalar::ONE)];
        let witness_times_root = PolynomialOps::mul(&w, root_monomial);

        //debug_assert!(vanishing_poly.len() == witness_times_root.len());
        for i in 0..vanishing_poly.len() {
            yield_constr.constraint_transition(vanishing_poly[i] - witness_times_root[i]);
        }
    }

    pub fn fpmul_ext_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        layout: FpMulLayout,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // get all the data
        let a = layout.a.evaluation_targets(&vars);
        let b = layout.b.evaluation_targets(&vars);
        let output = layout.output.evaluation_targets(&vars);

        let carry = layout.carry.evaluation_targets(&vars);
        let witness_low = layout.witness_low.evaluation_targets(&vars);
        let witness_high = layout.witness_high.evaluation_targets(&vars);

        // Construct the expected vanishing polynmial
        let ab = PolynomialGadget::mul_extension(builder, a, b);
        let ab_minus_output = PolynomialGadget::sub_extension(builder, &ab, output);
        let p_limbs =
            PolynomialGadget::constant_extension(builder, &P_iter().collect::<Vec<_>>()[..]);
        let mul_times_carry = PolynomialGadget::mul_extension(builder, carry, &p_limbs[..]);
        let vanishing_poly =
            PolynomialGadget::sub_extension(builder, &ab_minus_output, &mul_times_carry);

        // reconstruct witness

        // Reconstruct and shift back the witness polynomial
        let limb_const = F::Extension::from_canonical_u32(2u32.pow(16));
        let limb = builder.constant_extension(limb_const);
        let w_high_times_limb =
            PolynomialGadget::ext_scalar_mul_extension(builder, witness_high, &limb);
        let w_shifted = PolynomialGadget::add_extension(builder, witness_low, &w_high_times_limb);
        let offset =
            builder.constant_extension(F::Extension::from_canonical_u32(WITNESS_OFFSET as u32));
        let w = PolynomialGadget::sub_constant_extension(builder, &w_shifted, &offset);

        // Multiply by (x-2^16) and make the constraint
        let neg_limb = builder.constant_extension(-limb_const);
        let root_monomial = &[neg_limb, builder.constant_extension(F::Extension::ONE)];
        let witness_times_root =
            PolynomialGadget::mul_extension(builder, w.as_slice(), root_monomial);

        let constraint =
            PolynomialGadget::sub_extension(builder, &vanishing_poly, &witness_times_root);
        for constr in constraint {
            yield_constr.constraint_transition(builder, constr);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::arithmetic::arithmetic_stark::ArithmeticStark;
    use crate::arithmetic::builder::{ChipBuilder, TestStark};
    use crate::arithmetic::circuit::{EmulatedCircuitLayout, StarkParameters};
    use crate::arithmetic::eddsa::field::{Fp25519, Fp25519Param};
    use crate::arithmetic::trace::trace;
    use crate::arithmetic::InstructionT;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Copy, Debug)]
    pub struct FpMulLayoutCircuit;

    const LAYOUT: FpMulLayout = FpMulLayout::new(
        Register::Local(0, N_LIMBS),
        Register::Local(N_LIMBS, N_LIMBS),
        Register::Local(2 * N_LIMBS, N_LIMBS),
        Register::Local(3 * N_LIMBS, NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS),
    );

    const INLAYOUT: WriteInputLayout = WriteInputLayout::new(Register::Local(0, 2 * N_LIMBS));

    impl<F: RichField + Extendable<D>, const D: usize> EmulatedCircuitLayout<F, D, 2>
        for FpMulLayoutCircuit
    {
        const PUBLIC_INPUTS: usize = 0;
        const ENTRY_COLUMN: usize = 0;
        const NUM_ARITHMETIC_COLUMNS: usize = NUM_MUL_COLUMNS;
        const TABLE_INDEX: usize = NUM_MUL_COLUMNS;

        type Layouts = EpOpcodewithInputLayout;

        const OPERATIONS: [EpOpcodewithInputLayout; 2] = [
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::FpMul(LAYOUT)),
            EpOpcodewithInputLayout::Input(INLAYOUT),
        ];
    }

    #[derive(Debug, Clone)]
    pub struct FpMulInstruction {
        pub a: BigUint,
        pub b: BigUint,
    }

    impl FpMulInstruction {
        pub fn new(a: BigUint, b: BigUint) -> Self {
            Self { a, b }
        }
    }

    impl<F: RichField + Extendable<D>, const D: usize> InstructionT<FpMulLayoutCircuit, F, D, 2>
        for FpMulInstruction
    {
        fn generate_trace(self, pc: usize, tx: mpsc::Sender<(usize, usize, Vec<F>)>) {
            rayon::spawn(move || {
                let p_a = Polynomial::<F>::from_biguint_field(&self.a, 16, N_LIMBS);
                let p_b = Polynomial::<F>::from_biguint_field(&self.b, 16, N_LIMBS);

                let input = [p_a.into_vec(), p_b.into_vec()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();

                tx.send((pc, 1, input)).unwrap();
                let operation = EdOpcode::FpMul(self.a, self.b);
                let (trace_row, _) = operation.generate_trace_row();
                tx.send((pc, 0, trace_row)).unwrap();
            });
        }
    }

    #[test]
    fn test_arithmetic_stark_fpmul() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ArithmeticStark<FpMulLayoutCircuit, 2, F, D>;

        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = get_p();

        let mut rng = rand::thread_rng();

        let mut quad_operations = Vec::new();

        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(256) % &p22519;
            let b = rng.gen_biguint(256) % &p22519;

            quad_operations.push(FpMulInstruction::new(a, b));
        }

        let stark = S::new();

        let trace = stark.generate_trace(quad_operations);

        // Verify proof as a stark
        let proof =
            prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default()).unwrap();
        verify_stark_proof(stark, proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof =
            add_virtual_stark_proof_with_pis(&mut recursive_builder, stark, &config, degree_bits);

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

    #[derive(Clone, Debug, Copy)]
    struct FpMulTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for FpMulTest {
        const NUM_ARITHMETIC_COLUMNS: usize = NUM_MUL_COLUMNS;
        const PUBLIC_INPUTS: usize = 0;
        const NUM_STARK_COLUMNS: usize = 0;

        type Instruction = FpMull<Fp25519Param, 16>;
    }

    #[test]
    fn test_builder_fpmul() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type FpMul = FpMull<Fp25519Param, 16>;
        type S = TestStark<FpMulTest, F, D>;

        // build the stark
        let mut builder = ChipBuilder::<FpMulTest, F, D>::new();

        let a = builder.alloc_local::<Fp>().unwrap();
        let b = builder.alloc_local::<Fp>().unwrap();
        let result = builder.alloc_local::<Fp>().unwrap();

        let ab = FpMul::new(a, b, result);
        builder.insert_instruction(ab).unwrap();
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus_biguint();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            let handle = handle.clone();
            rayon::spawn(move || {
                let p_a = Polynomial::<F>::from_biguint_field(&a_int, 16, N_LIMBS);
                let p_b = Polynomial::<F>::from_biguint_field(&b_int, 16, N_LIMBS);

                handle.write_data(i as usize, a, p_a.into_vec()).unwrap();
                handle.write_data(i as usize, b, p_b.into_vec()).unwrap();

                let (row, _) = fpmul_trace::<F, D>(a_int, b_int);
                handle.write(i as usize, ab, row).unwrap();
            });
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows as usize).unwrap();

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
