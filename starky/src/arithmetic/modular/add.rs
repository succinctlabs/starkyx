//! Implementation of modular addition as a STARK (prototype)
//!
//! The implementation based on a method used in Polygon starks

use anyhow::{anyhow, Result};
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2_maybe_rayon::*;
use crate::arithmetic::builder::ChipBuilder;

use super::{ArithmeticParser, Register};
use crate::arithmetic::circuit::ChipParameters;
use crate::arithmetic::instruction::Instruction;
use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::arithmetic::register::{WitnessData, U16Array, DataRegister};
use crate::arithmetic::util::{self, to_field_iter};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub const N_LIMBS: usize = 16;

pub const NUM_INPUT_COLUMNS: usize = 2 * N_LIMBS;
pub const NUM_OUTPUT_COLUMNS: usize = N_LIMBS;
pub const NUM_MODULUS_COLUMNS: usize = N_LIMBS;
pub const NUM_CARRY_COLUMNS: usize = N_LIMBS;
pub const NUM_WTNESS_LOW_COLUMNS: usize = N_LIMBS - 1;
pub const NUM_WTNESS_HIGH_COLUMNS: usize = N_LIMBS - 1;
pub const NUM_ADD_WITNESS_COLUMNS: usize =
    NUM_CARRY_COLUMNS + NUM_WTNESS_LOW_COLUMNS + NUM_WTNESS_HIGH_COLUMNS;

pub const NUM_ARITH_COLUMNS: usize = 6 * N_LIMBS - 1 + N_LIMBS - 1;
pub const NUM_COLUMNS: usize = 1 + NUM_ARITH_COLUMNS + 2 * NUM_ARITH_COLUMNS;
pub const LOOKUP_SHIFT: usize = NUM_ARITH_COLUMNS + 1;
pub const RANGE_MAX: usize = 1usize << 16; // Range check strict upper bound
const WITNESS_OFFSET: usize = 1usize << 20; // Witness offset

#[inline]
pub const fn col_perm_index(i: usize) -> usize {
    2 * i + LOOKUP_SHIFT
}

#[inline]
pub const fn table_perm_index(i: usize) -> usize {
    2 * i + 1 + LOOKUP_SHIFT
}

#[derive(Debug, Clone, Copy)]
pub struct AddModLayout {
    input_1: Register,
    input_2: Register,
    output: Register,
    modulus: Register,
    carry: Register,
    witness_low: Register,
    witness_high: Register,
}

impl AddModLayout {
    pub const fn new(
        input_1: Register,
        input_2: Register,
        modulus: Register,
        output: Register,
        witness: Register,
    ) -> Self {
        /*debug_assert_eq!(input_1.len(), N_LIMBS);
        debug_assert_eq!(input_2.len(), N_LIMBS);
        debug_assert_eq!(modulus.len(), N_LIMBS);
        debug_assert_eq!(output.len(), N_LIMBS);
        debug_assert_eq!(
            witness.len(),
            NUM_CARRY_COLUMNS + NUM_WTNESS_LOW_COLUMNS + NUM_WTNESS_HIGH_COLUMNS
        );*/

        let (w_start, _) = witness.get_range();
        let carry = Register::Local(w_start, NUM_CARRY_COLUMNS);
        let witness_low = Register::Local(w_start + NUM_CARRY_COLUMNS, NUM_WTNESS_LOW_COLUMNS);
        let witness_high = Register::Local(
            w_start + NUM_CARRY_COLUMNS + NUM_WTNESS_LOW_COLUMNS,
            NUM_WTNESS_HIGH_COLUMNS,
        );

        Self {
            input_1,
            input_2,
            modulus,
            output,
            carry,
            witness_low,
            witness_high,
        }
    }

    #[inline]
    pub fn allocation_registers(&self) -> (Register, Register, Register) {
        let input = Register::Local(self.input_1.index(), 3 * N_LIMBS);
        let witness = Register::Local(self.carry.index(), NUM_ADD_WITNESS_COLUMNS);
        (input, self.output, witness)
    }

    #[inline]
    pub fn assign_row<T: Copy>(&self, trace_rows: &mut [Vec<T>], row: &mut [T], row_index: usize) {
        let (input_reg, output_reg, witness_reg) = self.allocation_registers();
        let input_slice = &mut row[0..3 * N_LIMBS];
        input_reg.assign(trace_rows, input_slice, row_index);
        let output_slice = &mut row[3 * N_LIMBS..4 * N_LIMBS];
        output_reg.assign(trace_rows, output_slice, row_index);
        let witness_slice = &mut row[4 * N_LIMBS..NUM_ARITH_COLUMNS];
        witness_reg.assign(trace_rows, witness_slice, row_index);
    }
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    /// Converts two BigUint inputs into the correspinding rows of addition mod modulus
    ///
    /// a + b = c mod m
    ///
    /// Each element represented by a polynomial a(x), b(x), c(x), m(x) of 16 limbs of 16 bits each
    /// We will witness the relation
    ///  a(x) + b(x) - c(x) - carry(x) * m(x) - (x - 2^16) * s(x) == 0
    /// a(x), b(x), c(x), m(x), s(x) should be range-checked.
    /// note carry = 0 or carry = 1
    pub fn add_trace(a: BigUint, b: BigUint, modulus: BigUint) -> Vec<F> {
        // Calculate all results as BigUint
        let result = (&a + &b) % &modulus;
        debug_assert!(result < modulus);
        let carry = (&a + &b - &result) / &modulus;
        debug_assert!(carry == BigUint::from(0u32) || carry == BigUint::from(1u32));

        // Make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(&a, 16, N_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(&b, 16, N_LIMBS);
        let p_m = Polynomial::<i64>::from_biguint_num(&modulus, 16, N_LIMBS);
        let p_res = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
        let p_c = Polynomial::<i64>::from_biguint_num(&carry, 16, N_LIMBS);
        let carry_bit = p_c.as_slice()[0];

        // Make the witness polynomial
        let vanishing_poly = &p_a + &p_b - &p_res - &p_m * carry_bit;
        debug_assert_eq!(vanishing_poly.degree(), N_LIMBS - 1);

        let witness_shifted =
            util::extract_witness_and_shift(&vanishing_poly, WITNESS_OFFSET as u32);
        let (witness_digit_low_f, witness_digit_high_f) = util::split_digits::<F>(&witness_shifted);

        // Make the row according to layout
        // input_1_index = 0;
        // input_2_index = N_LIMBS;
        // modulus_index = 2 * N_LIMBS;
        // output_index = 3 * N_LIMBS;
        // carry_index = 4 * N_LIMBS;
        // witness_index = 5 * N_LIMBS;
        let mut row = Vec::with_capacity(NUM_ARITH_COLUMNS);
        row.extend(to_field_iter::<F>(&p_a));
        row.extend(to_field_iter::<F>(&p_b));
        row.extend(to_field_iter::<F>(&p_m));
        row.extend(to_field_iter::<F>(&p_res));
        row.extend(to_field_iter::<F>(&p_c));
        row.extend(witness_digit_low_f);
        row.extend(witness_digit_high_f);

        // Allocate the values to the trace
        row
    }

    pub fn add_packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        layout: AddModLayout,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // Make polynomial limbs
        let a = layout.input_1.packed_entries_slice(&vars);
        let b = layout.input_2.packed_entries_slice(&vars);
        let m = layout.modulus.packed_entries_slice(&vars);
        let r = layout.output.packed_entries_slice(&vars);
        let c = layout.carry.packed_entries_slice(&vars);
        let w_low = layout.witness_low.packed_entries_slice(&vars);
        let w_high = layout.witness_high.packed_entries_slice(&vars);

        let limb: P = P::Scalar::from_canonical_u32(2u32.pow(16)).into();

        // Construct the vanishing polynomial
        let a_plus_b = PolynomialOps::add(a, b);
        let a_plus_b_minus_result = PolynomialOps::sub(&a_plus_b, r);
        let carry_times_mod = PolynomialOps::scalar_mul(m, &c[0]);
        let vanising_poly = PolynomialOps::sub(&a_plus_b_minus_result, &carry_times_mod);

        // Reconstruct and shift back the witness polynomial
        let w_shifted = w_low
            .iter()
            .zip(w_high.iter())
            .map(|(x, y)| *x + (*y * limb));

        let offset = FE::from_canonical_u32(WITNESS_OFFSET as u32);
        let w = w_shifted.map(|x| x - offset).collect::<Vec<P>>();

        // Multiply by (x-2^16) and make the constraint
        let root_monomial: &[P] = &[-limb, P::from(P::Scalar::ONE)];
        let witness_times_root = PolynomialOps::mul(&w, root_monomial);

        debug_assert!(vanising_poly.len() == witness_times_root.len());
        for i in 0..vanising_poly.len() {
            yield_constr.constraint_transition(vanising_poly[i] - witness_times_root[i]);
        }
    }

    pub fn add_ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        layout: AddModLayout,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // Make polynomial limbs
        let a = layout.input_1.evaluation_targets(&vars);
        let b = layout.input_2.evaluation_targets(&vars);
        let m = layout.modulus.evaluation_targets(&vars);
        let r = layout.output.evaluation_targets(&vars);
        let c = layout.carry.evaluation_targets(&vars);
        let w_low = layout.witness_low.evaluation_targets(&vars);
        let w_high = layout.witness_high.evaluation_targets(&vars);

        // Construct the vanishing polynomial
        let a_plus_b = PolynomialGadget::add_extension(builder, a, b);
        let a_plus_b_minus_result = PolynomialGadget::sub_extension(builder, &a_plus_b, r);
        let carry_times_mod = PolynomialGadget::ext_scalar_mul_extension(builder, m, &c[0]);
        let vanising_poly =
            PolynomialGadget::sub_extension(builder, &a_plus_b_minus_result, &carry_times_mod);

        // Reconstruct and shift back the witness polynomial
        let limb_const = F::Extension::from_canonical_u32(2u32.pow(16));
        let limb = builder.constant_extension(limb_const);
        let w_high_times_limb = PolynomialGadget::ext_scalar_mul_extension(builder, w_high, &limb);
        let w_shifted = PolynomialGadget::add_extension(builder, w_low, &w_high_times_limb);
        let offset =
            builder.constant_extension(F::Extension::from_canonical_u32(WITNESS_OFFSET as u32));
        let w = PolynomialGadget::sub_constant_extension(builder, &w_shifted, &offset);

        // Multiply by (x-2^16) and make the constraint
        let neg_limb = builder.constant_extension(-limb_const);
        let root_monomial = &[neg_limb, builder.constant_extension(F::Extension::ONE)];
        let witness_times_root =
            PolynomialGadget::mul_extension(builder, w.as_slice(), root_monomial);

        debug_assert!(vanising_poly.len() == witness_times_root.len());
        let constraint =
            PolynomialGadget::sub_extension(builder, &vanising_poly, &witness_times_root);
        for constr in constraint {
            yield_constr.constraint_transition(builder, constr);
        }
    }
}

type U256 = U16Array<N_LIMBS>;
#[derive(Debug, Clone, Copy)]
pub struct AddModInstruction {
    input_1: U256,
    input_2: U256,
    output: U256,
    modulus: U256,
    carry: Option<Register>,
    witness_low: Option<Register>,
    witness_high: Option<Register>,
}

impl AddModInstruction {
    pub fn new(input_1: U256, input_2: U256, output: U256, modulus: U256) -> Self {
        Self {
            input_1,
            input_2,
            output,
            modulus,
            carry: None,
            witness_low: None,
            witness_high: None,
        }
    }

    pub fn into_addmod_layout(&self) -> Result<AddModLayout> {
        let carry = self.carry.ok_or(anyhow!("missing carry"))?;
        let witness_low = self.witness_low.ok_or(anyhow!("missing witness_low"))?;
        let witness_high = self.witness_high.ok_or(anyhow!("missing witness_high"))?;
        Ok(AddModLayout {
            input_1: self.input_1.into_raw_register(),
            input_2: self.input_2.into_raw_register(),
            output: self.output.into_raw_register(),
            modulus: self.modulus.into_raw_register(),
            carry: carry,
            witness_low: witness_low,
            witness_high: witness_high,
        })
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for AddModInstruction {
    fn witness_data(&self) -> WitnessData {
        WitnessData::u16(NUM_ADD_WITNESS_COLUMNS)
    }

    fn shift_right(&mut self, _free_shift: usize, arithmetic_shift: usize) {
        let shift = arithmetic_shift;
        self.input_1.register_mut().shift_right(shift);
        self.input_2.register_mut().shift_right(shift);
        self.output.register_mut().shift_right(shift);
        self.modulus.register_mut().shift_right(shift);
        if let Some(carry) = self.carry.as_mut() {
            carry.shift_right(shift);
        }
        if let Some(witness_low) = self.witness_low.as_mut() {
            witness_low.shift_right(shift);
        }
        if let Some(witness_high) = self.witness_high.as_mut() {
            witness_high.shift_right(shift);
        }
    }

    fn set_witness(&mut self, register: Register) -> Result<()> {
        let (start, length) = (register.index(), register.len());
        if length != NUM_ADD_WITNESS_COLUMNS {
            return Err(anyhow!("Invalid witness length"));
        }
        let mut index = start;
        let carry = Register::Local(index, NUM_CARRY_COLUMNS);
        index += NUM_CARRY_COLUMNS;
        self.carry = Some(carry);
        let witness_low = Register::Local(index, NUM_WTNESS_LOW_COLUMNS);
        self.witness_low = Some(witness_low);
        index += NUM_WTNESS_LOW_COLUMNS;
        let witness_high = Register::Local(index, NUM_WTNESS_HIGH_COLUMNS);
        self.witness_high = Some(witness_high);
        Ok(())
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        let layout = self.into_addmod_layout().unwrap();
        layout.assign_row(trace_rows, row, row_index)
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
        let layout = self.into_addmod_layout().unwrap();
        ArithmeticParser::add_packed_generic_constraints(layout, vars, yield_constr);
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let layout = self.into_addmod_layout().unwrap();
        ArithmeticParser::add_ext_circuit(layout, builder, vars, yield_constr);
    }
}



impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {}


pub struct Bye<F: RichField + Extendable<D>, const D: usize>  {
    a : Vec<Box<dyn FnOnce(&mut [Vec<F>], &mut [F], usize)>>
}

impl AddModInstruction {
    fn make_assign<F : Copy>(&self) -> impl FnOnce(&mut [Vec<F>], &mut [F], usize) {
        let layout = self.into_addmod_layout().unwrap();
        move |trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize| {
            layout.assign_row(trace_rows, row, row_index)
        }
    }
}

impl <F: RichField + Extendable<D>, const D: usize>  Bye<F, D> {
    pub fn new() -> Self {
        Self {
            a : vec![]
        }
    }

    pub fn add(&mut self, instruction : AddModInstruction) {
        self.a.push(Box::new(instruction.make_assign()));
    }
}