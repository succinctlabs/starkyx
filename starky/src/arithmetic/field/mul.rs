use anyhow::{anyhow, Result};
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::*;
use crate::arithmetic::builder::ChipBuilder;
use crate::arithmetic::chip::ChipParameters;
use crate::arithmetic::instruction::Instruction;
use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::arithmetic::register::{DataRegister, Register, WitnessData};
use crate::arithmetic::trace::TraceHandle;
use crate::arithmetic::util::{extract_witness_and_shift, split_digits, to_field_iter};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub struct FpMul<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> {
    a: FieldRegister<P, N_LIMBS>,
    b: FieldRegister<P, N_LIMBS>,
    result: FieldRegister<P, N_LIMBS>,
    carry: Option<Register>,
    witness_low: Option<Register>,
    witness_high: Option<Register>,
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
    pub fn fpmul<P: FieldParameters<N>, const N: usize>(
        &mut self,
        a: &FieldRegister<P, N>,
        b: &FieldRegister<P, N>,
        result: &FieldRegister<P, N>,
    ) -> Result<FpMul<P, N>>
    where
        L::Instruction: From<FpMul<P, N>>,
    {
        let instr = FpMul::new(*a, *b, *result);
        self.insert_instruction(instr.into())?;
        Ok(instr)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FpMulConst<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> {
    a: FieldRegister<P, N_LIMBS>,
    c: [u16; N_LIMBS],
    result: FieldRegister<P, N_LIMBS>,
    carry: Option<Register>,
    witness_low: Option<Register>,
    witness_high: Option<Register>,
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
    pub fn fpmul_const<P: FieldParameters<N>, const N: usize>(
        &mut self,
        a: &FieldRegister<P, N>,
        c: [u16; N],
        result: &FieldRegister<P, N>,
    ) -> Result<FpMulConst<P, N>>
    where
        L::Instruction: From<FpMulConst<P, N>>,
    {
        let instr = FpMulConst::new(*a, c, *result);
        self.insert_instruction(instr.into())?;
        Ok(instr)
    }
}

impl<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> FpMul<P, N_LIMBS> {
    const NUM_CARRY_LIMBS: usize = N_LIMBS;
    pub const NUM_WITNESS_LOW_LIMBS: usize = 2 * N_LIMBS - 2;
    pub const NUM_WITNESS_HIGH_LIMBS: usize = 2 * N_LIMBS - 2;

    #[inline]
    pub const fn new(
        a: FieldRegister<P, N_LIMBS>,
        b: FieldRegister<P, N_LIMBS>,
        result: FieldRegister<P, N_LIMBS>,
    ) -> Self {
        Self {
            a,
            b,
            result,
            carry: None,
            witness_low: None,
            witness_high: None,
        }
    }

    pub const fn num_mul_columns() -> usize {
        3 * N_LIMBS
            + Self::NUM_CARRY_LIMBS
            + Self::NUM_WITNESS_LOW_LIMBS
            + Self::NUM_WITNESS_HIGH_LIMBS
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const N: usize, FP: FieldParameters<N>>
    Instruction<F, D> for FpMul<FP, N>
{
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
            _ => return Err(anyhow!("Invalid witness register")),
        };
        self.carry = Some(carry);
        self.witness_low = Some(witness_low);
        self.witness_high = Some(witness_high);

        Ok(())
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        let mut index = 0;
        self.result
            .register()
            .assign(trace_rows, &mut row[index..N], row_index);
        index += N;
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
        let a = self.a.register().packed_entries(&vars);
        let b = self.b.register().packed_entries(&vars);
        let result = self.result.register().packed_entries(&vars);

        let carry = self.carry.unwrap().packed_entries_slice(&vars);
        let witness_low = self.witness_low.unwrap().packed_entries_slice(&vars);
        let witness_high = self.witness_high.unwrap().packed_entries_slice(&vars);

        // Construct the expected vanishing polynmial
        let ab = PolynomialOps::mul(&a, &b);
        let ab_minus_result = PolynomialOps::sub(&ab, &result);
        let p_limbs = Polynomial::<FE>::from_iter(modulus_field_iter::<FE, FP, N>());
        let mul_times_carry = PolynomialOps::scalar_poly_mul(carry, p_limbs.as_slice());
        let vanishing_poly = PolynomialOps::sub(&ab_minus_result, &mul_times_carry);

        // reconstruct witness

        let limb = FE::from_canonical_u32(LIMB);

        // Reconstruct and shift back the witness polynomial
        let w_shifted = witness_low
            .iter()
            .zip(witness_high.iter())
            .map(|(x, y)| *x + (*y * limb));

        let offset = FE::from_canonical_u32(FP::WITNESS_OFFSET as u32);
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
        let p_limbs = PolynomialGadget::constant_extension(
            builder,
            &modulus_field_iter::<F::Extension, FP, N>().collect::<Vec<_>>(),
        );
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
            builder.constant_extension(F::Extension::from_canonical_u32(FP::WITNESS_OFFSET as u32));
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

impl<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> FpMul<P, N_LIMBS> {
    /// Trace row for fp_mul operation
    ///
    /// Returns a vector
    /// [Input[2 * N_LIMBS], output[N_LIMBS], carry[NUM_CARRY_LIMBS], Witness_low[NUM_WITNESS_LIMBS], Witness_high[NUM_WITNESS_LIMBS]]
    pub fn trace_row<F: RichField + Extendable<D>, const D: usize>(
        a: &BigUint,
        b: &BigUint,
    ) -> (Vec<F>, BigUint) {
        let p = P::modulus_biguint();
        let result = (a * b) % &p;
        debug_assert!(result < p);
        let carry = (a * b - &result) / &p;
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, a * b - &result);

        // make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(a, 16, N_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(b, 16, N_LIMBS);
        let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, N_LIMBS);

        let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
        let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, Self::NUM_CARRY_LIMBS);

        // Compute the vanishing polynomial
        let vanishing_poly = &p_a * &p_b - &p_result - &p_carry * &p_p;
        debug_assert_eq!(
            vanishing_poly.degree(),
            Self::NUM_WITNESS_LOW_LIMBS + Self::NUM_WITNESS_HIGH_LIMBS
        );

        // Compute the witness
        let witness_shifted = extract_witness_and_shift(&vanishing_poly, P::WITNESS_OFFSET as u32);
        let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

        let mut row = Vec::with_capacity(Self::num_mul_columns());

        // output
        row.extend(to_field_iter::<F>(&p_result));
        // carry and witness
        row.extend(to_field_iter::<F>(&p_carry));
        row.extend(witness_low);
        row.extend(witness_high);

        (row, result)
    }

    pub fn write<F: RichField + Extendable<D>, const D: usize>(
        &self,
        a: BigUint,
        b: BigUint,
        handle: TraceHandle<F, D>,
        row_index: usize,
    ) -> Result<()> {
        let (row, _) = Self::trace_row::<F, D>(&a, &b);
        handle.write(row_index, *self, row)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceHandle<F, D> {
    pub fn write_fpmul<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize>(
        &self,
        row_index: usize,
        a_int: &BigUint,
        b_int: &BigUint,
        instruction: FpMul<P, N_LIMBS>,
    ) -> Result<BigUint> {
        let (row, result) = FpMul::<P, N_LIMBS>::trace_row::<F, D>(a_int, b_int);
        self.write(row_index, instruction, row)?;
        Ok(result)
    }
}

impl<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> FpMulConst<P, N_LIMBS> {
    const NUM_CARRY_LIMBS: usize = N_LIMBS;
    pub const NUM_WITNESS_LOW_LIMBS: usize = 2 * N_LIMBS - 2;
    pub const NUM_WITNESS_HIGH_LIMBS: usize = 2 * N_LIMBS - 2;

    #[inline]
    pub const fn new(
        a: FieldRegister<P, N_LIMBS>,
        c: [u16; N_LIMBS],
        result: FieldRegister<P, N_LIMBS>,
    ) -> Self {
        Self {
            a,
            c,
            result,
            carry: None,
            witness_low: None,
            witness_high: None,
        }
    }

    pub const fn num_mul_const_columns() -> usize {
        2 * N_LIMBS
            + Self::NUM_CARRY_LIMBS
            + Self::NUM_WITNESS_LOW_LIMBS
            + Self::NUM_WITNESS_HIGH_LIMBS
    }
}

// ------------------------------------------------------------
// Fp Mul const implementation
// ------------------------------------------------------------

impl<F: RichField + Extendable<D>, const D: usize, const N: usize, FP: FieldParameters<N>>
    Instruction<F, D> for FpMulConst<FP, N>
{
    fn memory_vec(&self) -> Vec<Register> {
        vec![*self.a.register(), *self.result.register()]
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
            _ => return Err(anyhow!("Invalid witness register")),
        };
        self.carry = Some(carry);
        self.witness_low = Some(witness_low);
        self.witness_high = Some(witness_high);

        Ok(())
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        let mut index = 0;
        self.result
            .register()
            .assign(trace_rows, &mut row[index..N], row_index);
        index += N;
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
        let a = self.a.register().packed_entries(&vars);
        let c = self
            .c
            .into_iter()
            .map(FE::from_canonical_u16)
            .map(P::from)
            .collect::<Vec<_>>();
        let result = self.result.register().packed_entries(&vars);

        let carry = self.carry.unwrap().packed_entries_slice(&vars);
        let witness_low = self.witness_low.unwrap().packed_entries_slice(&vars);
        let witness_high = self.witness_high.unwrap().packed_entries_slice(&vars);

        // Construct the expected vanishing polynmial
        let ac = PolynomialOps::mul(&a, &c);
        let ac_minus_result = PolynomialOps::sub(&ac, &result);
        let p_limbs = Polynomial::<FE>::from_iter(modulus_field_iter::<FE, FP, N>());
        let mul_times_carry = PolynomialOps::scalar_poly_mul(carry, p_limbs.as_slice());
        let vanishing_poly = PolynomialOps::sub(&ac_minus_result, &mul_times_carry);

        // reconstruct witness
        let limb = FE::from_canonical_u32(LIMB);

        // Reconstruct and shift back the witness polynomial
        let w_shifted = witness_low
            .iter()
            .zip(witness_high.iter())
            .map(|(x, y)| *x + (*y * limb));

        let offset = FE::from_canonical_u32(FP::WITNESS_OFFSET as u32);
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
        let c_vec = self
            .c
            .into_iter()
            .map(F::Extension::from_canonical_u16)
            .collect::<Vec<_>>();
        let c = PolynomialGadget::constant_extension(builder, &c_vec);
        let result = self.result.register().evaluation_targets(&vars);

        let carry = self.carry.unwrap().evaluation_targets(&vars);
        let witness_low = self.witness_low.unwrap().evaluation_targets(&vars);
        let witness_high = self.witness_high.unwrap().evaluation_targets(&vars);

        // Construct the expected vanishing polynmial
        let ac = PolynomialGadget::mul_extension(builder, a, &c);
        let ac_minus_result = PolynomialGadget::sub_extension(builder, &ac, result);
        let p_limbs = PolynomialGadget::constant_extension(
            builder,
            &modulus_field_iter::<F::Extension, FP, N>().collect::<Vec<_>>(),
        );
        let mul_times_carry = PolynomialGadget::mul_extension(builder, carry, &p_limbs[..]);
        let vanishing_poly =
            PolynomialGadget::sub_extension(builder, &ac_minus_result, &mul_times_carry);

        // reconstruct witness

        // Reconstruct and shift back the witness polynomial
        let limb_const = F::Extension::from_canonical_u32(2u32.pow(16));
        let limb = builder.constant_extension(limb_const);
        let w_high_times_limb =
            PolynomialGadget::ext_scalar_mul_extension(builder, witness_high, &limb);
        let w_shifted = PolynomialGadget::add_extension(builder, witness_low, &w_high_times_limb);
        let offset =
            builder.constant_extension(F::Extension::from_canonical_u32(FP::WITNESS_OFFSET as u32));
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

impl<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize> FpMulConst<P, N_LIMBS> {
    /// Trace row for fp_mul operation
    ///
    /// Returns a vector
    /// [Input[2 * N_LIMBS], output[N_LIMBS], carry[NUM_CARRY_LIMBS], Witness_low[NUM_WITNESS_LIMBS], Witness_high[NUM_WITNESS_LIMBS]]
    pub fn trace_row<F: RichField + Extendable<D>, const D: usize>(
        &self,
        a: &BigUint,
    ) -> (Vec<F>, BigUint) {
        let p = P::modulus_biguint();
        let mut c = BigUint::zero();
        for (i, limb) in self.c.iter().enumerate() {
            c += BigUint::from(*limb) << (16 * i);
        }
        let result = (a * &c) % &p;
        debug_assert!(result < p);
        let carry = (a * &c - &result) / &p;
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, a * &c - &result);

        // make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(a, 16, N_LIMBS);
        let p_c = Polynomial::<i64>::from_biguint_num(&c, 16, N_LIMBS);
        let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, N_LIMBS);

        let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
        let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, Self::NUM_CARRY_LIMBS);

        // Compute the vanishing polynomial
        let vanishing_poly = &p_a * &p_c - &p_result - &p_carry * &p_p;
        debug_assert_eq!(
            vanishing_poly.degree(),
            Self::NUM_WITNESS_LOW_LIMBS + Self::NUM_WITNESS_HIGH_LIMBS
        );

        // Compute the witness
        let witness_shifted = extract_witness_and_shift(&vanishing_poly, P::WITNESS_OFFSET as u32);
        let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

        let mut row = Vec::with_capacity(Self::num_mul_const_columns());

        // output
        row.extend(to_field_iter::<F>(&p_result));
        // carry and witness
        row.extend(to_field_iter::<F>(&p_carry));
        row.extend(witness_low);
        row.extend(witness_high);

        (row, result)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceHandle<F, D> {
    pub fn write_fpmul_const<P: FieldParameters<N_LIMBS>, const N_LIMBS: usize>(
        &self,
        row_index: usize,
        a_int: &BigUint,
        instruction: FpMulConst<P, N_LIMBS>,
    ) -> Result<BigUint> {
        let (row, result) = instruction.trace_row::<F, D>(a_int);
        self.write(row_index, instruction, row)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::arithmetic::builder::ChipBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
    use crate::arithmetic::field::Fp25519Param;
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    struct FpMulTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for FpMulTest {
        const NUM_ARITHMETIC_COLUMNS: usize = FpMul::<Fp25519Param, 16>::num_mul_columns();
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = FpMul<Fp25519Param, 16>;
    }

    #[test]
    fn test_fpmul_row() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<FpMulTest, F, D>;

        // build the stark
        let mut builder = ChipBuilder::<FpMulTest, F, D>::new();

        let a = builder.alloc_local::<Fp>().unwrap();
        let b = builder.alloc_local::<Fp>().unwrap();
        let result = builder.alloc_local::<Fp>().unwrap();

        //let ab = FMul::new(a, b, result);
        //builder.insert_instruction(ab).unwrap();
        let ab_ins = builder.fpmul(&a, &b, &result).unwrap();
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus_biguint();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            let handle = handle.clone();
            rayon::spawn(move || {
                handle.write_field(i, &a_int, a).unwrap();
                handle.write_field(i, &b_int, b).unwrap();
                handle.write_fpmul(i, &a_int, &b_int, ab_ins).unwrap();
            });
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

    #[test]
    fn test_fpmul_multi_row() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<FpMulTest, F, D>;

        // build the stark
        let mut builder = ChipBuilder::<FpMulTest, F, D>::new();

        let a = builder.alloc_local::<Fp>().unwrap();
        let b = builder.alloc_local::<Fp>().unwrap();
        let result = builder.alloc_next::<Fp>().unwrap();
        let Register::Local(index, length) = *a.register() else {
            panic!("a is not a local register");
        };
        assert_eq!(Register::Next(index, length), *result.register());

        //let ab = FMul::new(a, b, result);
        //builder.insert_instruction(ab).unwrap();
        let mul_transition = builder.fpmul(&a, &b, &result).unwrap();
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus_biguint();

        let mut rng = thread_rng();
        let a_int_init: BigUint = rng.gen_biguint(256) % &p;
        let b_int_init = rng.gen_biguint(256) % &p;
        handle.write_field(0, &a_int_init, a).unwrap();
        handle.write_field(0, &b_int_init, b).unwrap();
        let mut a_int = handle
            .write_fpmul(0, &a_int_init, &b_int_init, mul_transition)
            .unwrap();
        for i in 1..num_rows - 1 {
            let b_int: BigUint = rng.gen_biguint(256) % &p;
            handle.write_field(i as usize, &b_int, b).unwrap();
            a_int = handle
                .write_fpmul(i as usize, &a_int, &b_int, mul_transition)
                .unwrap();
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

    #[derive(Clone, Debug, Copy)]
    struct FpMulConstTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for FpMulConstTest {
        const NUM_ARITHMETIC_COLUMNS: usize =
            FpMulConst::<Fp25519Param, 16>::num_mul_const_columns();
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = FpMulConst<Fp25519Param, 16>;
    }

    #[test]
    fn test_fpmul_const() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<FpMulConstTest, F, D>;

        let c: [u16; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        // build the stark
        let mut builder = ChipBuilder::<FpMulConstTest, F, D>::new();

        let a = builder.alloc_local::<Fp>().unwrap();
        let result = builder.alloc_local::<Fp>().unwrap();

        //let ab = FMul::new(a, b, result);
        //builder.insert_instruction(ab).unwrap();
        let ac_ins = builder.fpmul_const(&a, c, &result).unwrap();
        builder.write_data(&a).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus_biguint();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let handle = handle.clone();
            rayon::spawn(move || {
                handle.write_field(i, &a_int, a).unwrap();
                let res = handle.write_fpmul_const(i, &a_int, ac_ins).unwrap();
                assert_eq!(res, a_int);
            });
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
