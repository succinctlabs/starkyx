//! Implementation of modular addition as a STARK (prototype)
//!
//! The implementation based on a method used in Polygon starks

use alloc::vec;
use core::marker::PhantomData;
use std::sync::mpsc;

use num::{BigInt, BigUint};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::util::transpose;
use plonky2_maybe_rayon::*;

use super::{ArithmeticOp, ArithmeticParser, Register};
use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::lookup::{eval_lookups, eval_lookups_circuit, permuted_cols};
use crate::permutation::PermutationPair;
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub const N_LIMBS: usize = 16;
pub const NUM_INPUT_COLUMNS: usize = 2 * N_LIMBS;
pub const NUM_OUTPUT_COLUMNS: usize = N_LIMBS;
pub const NUM_MODULUS_COLUMNS: usize = N_LIMBS;
pub const NUM_CARRY_COLUMNS: usize = N_LIMBS;
pub const NUM_WITNESS_COLUMNS: usize = 2 * N_LIMBS - 2;

pub const NUM_ARITH_COLUMNS: usize = NUM_INPUT_COLUMNS
    + NUM_OUTPUT_COLUMNS
    + NUM_MODULUS_COLUMNS
    + NUM_CARRY_COLUMNS
    + 2 * NUM_WITNESS_COLUMNS;
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
pub struct MulModLayout {
    input_1: Register,
    input_2: Register,
    output: Register,
    modulus: Register,
    carry: Register,
    witness_low: Register,
    witness_high: Register,
}

impl MulModLayout {
    pub fn new(
        input_1: Register,
        input_2: Register,
        modulus: Register,
        output: Register,
        witness : Register,
    ) -> Self {
        debug_assert_eq!(input_1.len(), N_LIMBS);
        debug_assert_eq!(input_2.len(), N_LIMBS);
        debug_assert_eq!(output.len(), NUM_OUTPUT_COLUMNS);
        debug_assert_eq!(modulus.len(), NUM_MODULUS_COLUMNS);
        debug_assert_eq!(witness.len(), NUM_CARRY_COLUMNS + 2 * NUM_WITNESS_COLUMNS);

        let (witness_start, _) = witness.get_range();
        let carry = Register::Local(witness_start, NUM_CARRY_COLUMNS);
        let witness_low = Register::Local(witness_start + NUM_CARRY_COLUMNS, NUM_WITNESS_COLUMNS);
        let witness_high = Register::Local(witness_start + NUM_CARRY_COLUMNS + NUM_WITNESS_COLUMNS, NUM_WITNESS_COLUMNS);

        Self {
            input_1,
            input_2,
            output,
            modulus,
            carry,
            witness_low,
            witness_high,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MulStark<F, const D: usize> {
    layout: MulModLayout,
    _marker: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    /// Converts two BigUint inputs into the correspinding rows of multiplication mod modulus
    ///
    /// a * b = c mod m
    ///
    /// Each element represented by a polynomial a(x), b(x), c(x), m(x) of 16 limbs of 16 bits each
    /// We will witness the relation
    ///  a(x) * b(x) - c(x) - carry(x) * m(x) - (x - 2^16) * s(x) == 0
    /// a(x), b(x), c(x), m(x), s(x) should be range-checked.
    pub fn mul_trace(a: BigUint, b: BigUint, modulus: BigUint) -> Vec<F> {
        let result = (&a * &b) % &modulus;
        debug_assert!(result < modulus);

        let carry = (&a * &b - &result) / &modulus;
        debug_assert!(carry < modulus);

        // make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(&a, 16, N_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(&b, 16, N_LIMBS);
        let p_m = Polynomial::<i64>::from_biguint_num(&modulus, 16, N_LIMBS);
        let p_res = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
        let p_c = Polynomial::<i64>::from_biguint_num(&carry, 16, N_LIMBS);

        // Compute the witness polynomial
        let vanishing_poly = &p_a * &p_b - &p_res - &p_c * &p_m;
        debug_assert_eq!(vanishing_poly.degree(), 2 * N_LIMBS - 2);

        //sanity check
        let eval_vanishing = vanishing_poly
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, x)| BigInt::from(2u32).pow(16 * i as u32) * x)
            .sum::<BigInt>();
        debug_assert_eq!(eval_vanishing, BigInt::from(0));

        // extract witness from vanishing polynomial
        let limb = 2u32.pow(16) as i64;
        let witness_poly = vanishing_poly.root_quotient(limb);
        assert_eq!(witness_poly.degree(), 2 * N_LIMBS - 3);

        for c in witness_poly.as_slice().iter() {
            debug_assert!(c.abs() < WITNESS_OFFSET as i64);
        }

        // Sanity check
        debug_assert_eq!(
            (&witness_poly * &(Polynomial::<i64>::new_from_slice(&[-limb, 1]))).as_slice(),
            vanishing_poly.as_slice()
        );

        // Shifting the witness polynomial to make it positive
        let witness_poly_shifted_iter = witness_poly
            .coefficients()
            .iter()
            .map(|x| x + WITNESS_OFFSET as i64)
            .map(|x| u32::try_from(x).unwrap())
            .collect::<Vec<_>>();

        // Store each of the witness u32 digits as two u16 digits
        let witness_digit_high_f = witness_poly_shifted_iter
            .iter()
            .map(|x| (*x >> 16) as u16)
            .map(|x| F::from_canonical_u16(x));
        let witness_digit_low_f = witness_poly_shifted_iter
            .iter()
            .map(|x| *x as u16)
            .map(|x| F::from_canonical_u16(x));

        let p_a_f = p_a
            .coefficients()
            .into_iter()
            .map(|x| F::from_canonical_u32(x as u32));
        let p_b_f = p_b
            .coefficients()
            .into_iter()
            .map(|x| F::from_canonical_u32(x as u32));
        let p_m_f = p_m
            .coefficients()
            .into_iter()
            .map(|x| F::from_canonical_u32(x as u32));
        let p_res_f = p_res
            .coefficients()
            .into_iter()
            .map(|x| F::from_canonical_u32(x as u32));
        let p_c_f = p_c
            .coefficients()
            .into_iter()
            .map(|x| F::from_canonical_u32(x as u32));

        // Make the row according to layout
        // input_1_index = 0;
        // input_2_index = N_LIMBS;
        // modulus_index = 2 * N_LIMBS;
        // output_index = 3 * N_LIMBS;
        // carry_index = 4 * N_LIMBS;
        // witness_index = 5 * N_LIMBS;
        let mut row = Vec::with_capacity(NUM_ARITH_COLUMNS);
        row.extend(p_a_f);
        row.extend(p_b_f);
        row.extend(p_m_f);
        row.extend(p_res_f);
        row.extend(p_c_f);
        row.extend(witness_digit_low_f);
        row.extend(witness_digit_high_f);

        // Allocate the values to the trace
        row
    }

    pub fn mul_packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        layout: MulModLayout,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // Get the range of the variables
        let (a_l, a_h) = layout.input_1.get_range();
        let (b_l, b_h) = layout.input_2.get_range();
        let (r_l, r_h) = layout.output.get_range();
        let (m_l, m_h) = layout.modulus.get_range();
        let (c_l, c_h) = layout.carry.get_range();
        let (w_low_l, w_low_h) = layout.witness_low.get_range();
        let (w_high_l, w_high_h) = layout.witness_high.get_range();

        // Make polynomial limbs
        let a = &vars.local_values[a_l..a_h];
        let b = &vars.local_values[b_l..b_h];
        let m = &vars.local_values[m_l..m_h];
        let r = &vars.local_values[r_l..r_h];
        let c = &vars.local_values[c_l..c_h];
        let w_low = &vars.local_values[w_low_l..w_low_h];
        let w_high = &vars.local_values[w_high_l..w_high_h];

        let limb: P = P::Scalar::from_canonical_u32(2u32.pow(16)).into();

        // Construct the vanishing polynomial
        let a_mul_b = PolynomialOps::mul(a, b);
        let a_mul_b_minus_result = PolynomialOps::sub(&a_mul_b, r);
        let carry_times_mod = PolynomialOps::mul(m, c);
        let vanising_poly = PolynomialOps::sub(&a_mul_b_minus_result, &carry_times_mod);

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

    pub fn mul_ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        layout: MulModLayout,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // Get the range of the variables
        let (a_l, a_h) = layout.input_1.get_range();
        let (b_l, b_h) = layout.input_2.get_range();
        let (r_l, r_h) = layout.output.get_range();
        let (m_l, m_h) = layout.modulus.get_range();
        let (c_l, c_h) = layout.carry.get_range();
        let (w_low_l, w_low_h) = layout.witness_low.get_range();
        let (w_high_l, w_high_h) = layout.witness_high.get_range();

        // Make polynomial limbs
        let a = &vars.local_values[a_l..a_h];
        let b = &vars.local_values[b_l..b_h];
        let m = &vars.local_values[m_l..m_h];
        let r = &vars.local_values[r_l..r_h];
        let c = &vars.local_values[c_l..c_h];
        let w_low = &vars.local_values[w_low_l..w_low_h];
        let w_high = &vars.local_values[w_high_l..w_high_h];

        // Construct the vanishing polynomial
        let a_mul_b = PolynomialGadget::mul_extension(builder, a, b);
        let a_mul_b_minus_result = PolynomialGadget::sub_extension(builder, &a_mul_b, r);
        let carry_times_mod = PolynomialGadget::mul_extension(builder, m, c);
        let vanising_poly =
            PolynomialGadget::sub_extension(builder, &a_mul_b_minus_result, &carry_times_mod);

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

impl<F: RichField + Extendable<D>, const D: usize> MulStark<F, D> {
    fn gen_trace(&self, program: Vec<ArithmeticOp>) -> Vec<PolynomialValues<F>> {
        let num_operations = program.len();
        let num_rows = num_operations;
        assert_eq!(num_rows, 1 << 16);
        let mut trace_rows = vec![Vec::with_capacity(NUM_ARITH_COLUMNS); num_rows];
        //let mut trace_cols = vec![vec![F::ZERO;num_rows]; NUM_COLUMNS];

        let (tx, rx) = mpsc::channel::<(usize, Vec<F>)>();

        for (i, op) in program.into_iter().enumerate() {
            let tx = tx.clone();
            let ArithmeticOp::MulMod(a, b, m) = op else { panic!("Invalid op") };
            //rayon::spawn(move || {
            let mut row = ArithmeticParser::<F, D>::mul_trace(a, b, m);
            debug_assert!(row.len() == NUM_ARITH_COLUMNS);
            row.push(F::from_canonical_usize(i));
            tx.send((i, row)).unwrap();
            //});
        }
        drop(tx);

        // Collecte the trace rows which are processed in parallel
        while let Ok((i, mut row)) = rx.recv() {
            trace_rows[i].append(&mut row); // Append row to trace
        }

        debug_assert_eq!(trace_rows.len(), num_rows);

        // Transpose the trace to get the columns and resize to the correct size
        let mut trace_cols = transpose(&trace_rows);
        trace_cols.resize(NUM_COLUMNS, Vec::with_capacity(num_rows));

        // Calculate the permutation and append permuted columbs to trace
        let (trace_values, perm_values) = trace_cols.split_at_mut(NUM_ARITH_COLUMNS + 1);
        (0..NUM_ARITH_COLUMNS)
            .into_par_iter()
            .map(|i| permuted_cols(&trace_values[i], &trace_values[NUM_ARITH_COLUMNS]))
            .zip(perm_values.par_iter_mut().chunks(2))
            .for_each(|((col_perm, table_perm), mut trace)| {
                trace[0].extend(col_perm);
                trace[1].extend(table_perm);
            });

        debug_assert_eq!(trace_cols.len(), NUM_COLUMNS);
        debug_assert_eq!(trace_cols[0].len(), num_rows);

        trace_cols.into_iter().map(PolynomialValues::new).collect()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MulStark<F, D> {
    const COLUMNS: usize = NUM_COLUMNS;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        ArithmeticParser::mul_packed_generic_constraints(self.layout, vars, yield_constr);
        // lookp table values
        // lookp table values
        yield_constr.constraint_first_row(vars.local_values[NUM_ARITH_COLUMNS]);
        let table_values_relation =
            vars.local_values[NUM_ARITH_COLUMNS] + FE::ONE - vars.next_values[NUM_ARITH_COLUMNS];
        yield_constr.constraint_transition(table_values_relation);
        // permutations
        for i in 0..NUM_ARITH_COLUMNS {
            eval_lookups(vars, yield_constr, col_perm_index(i), table_perm_index(i));
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        ArithmeticParser::mul_ext_circuit(self.layout, builder, vars, yield_constr);
        // lookup table values
        yield_constr.constraint_first_row(builder, vars.local_values[NUM_ARITH_COLUMNS]);
        let one = builder.constant_extension(F::Extension::ONE);
        let table_plus_one = builder.add_extension(vars.local_values[NUM_ARITH_COLUMNS], one);
        let table_relation =
            builder.sub_extension(table_plus_one, vars.next_values[NUM_ARITH_COLUMNS]);
        yield_constr.constraint_transition(builder, table_relation);

        // lookup argument
        for i in 0..NUM_ARITH_COLUMNS {
            eval_lookups_circuit(
                builder,
                vars,
                yield_constr,
                col_perm_index(i),
                table_perm_index(i),
            );
        }
    }

    fn constraint_degree(&self) -> usize {
        2
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        (0..NUM_ARITH_COLUMNS)
            .flat_map(|i| {
                [
                    PermutationPair::singletons(i, col_perm_index(i)),
                    PermutationPair::singletons(NUM_ARITH_COLUMNS, table_perm_index(i)),
                ]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[test]
    fn test_arithmetic_op_mul() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = MulStark<F, D>;

        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        let mut multiplications = Vec::new();

        let input_1_index = 0;
        let input_2_index = N_LIMBS;
        let modulus_index = 2 * N_LIMBS;
        let output_index = 3 * N_LIMBS;

        let layout = MulModLayout::new(
            Register::Local(input_1_index, N_LIMBS),
            Register::Local(input_2_index, N_LIMBS),
            Register::Local(modulus_index, N_LIMBS),
            Register::Local(output_index, NUM_OUTPUT_COLUMNS),
            Register::Local(4 * N_LIMBS, NUM_CARRY_COLUMNS + NUM_WITNESS_COLUMNS)
        );

        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(255) % &p22519;
            let b = rng.gen_biguint(255) % &p22519;
            let p = p22519.clone();

            let operation = ArithmeticOp::MulMod(a.clone(), b.clone(), p.clone());
            multiplications.push(operation);
        }

        let stark = S {
            layout,
            _marker: PhantomData,
        };

        let trace = stark.gen_trace(multiplications);

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
