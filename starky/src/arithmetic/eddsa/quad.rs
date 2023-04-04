use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::*;
use crate::arithmetic::polynomial::{Polynomial, PolynomialOps};
use crate::arithmetic::util::{extract_witness_and_shift, split_digits, to_field_iter};
use crate::arithmetic::{ArithmeticParser, Register};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub const N_LIMBS: usize = 16;
pub const NUM_CARRY_LIMBS: usize = N_LIMBS;
pub const NUM_WITNESS_LIMBS: usize = 2 * N_LIMBS - 3;
const WITNESS_OFFSET: usize = 1usize << 20; // Witness offset
const NUM_QUAD_COLUMNS: usize = 5 * N_LIMBS + NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS;

/// A gadget to compute
/// QUAD(x, y, z, w) = (a * b + c * d) mod p
#[derive(Debug, Clone, Copy)]
pub struct QuadLayout {
    a: Register,
    b: Register,
    c: Register,
    d: Register,
    output: Register,
    witness: Register,
    carry: Register,
    witness_low: Register,
    witness_high: Register,
}

impl QuadLayout {
    #[inline]
    pub const fn new(
        a: Register,
        b: Register,
        c: Register,
        d: Register,
        output: Register,
        witness: Register,
    ) -> Self {
        let (carry, witness_low, witness_high) = match witness {
            Register::Local(index, length) => (
                Register::Local(index, NUM_CARRY_LIMBS),
                Register::Local(index + NUM_CARRY_LIMBS, NUM_WITNESS_LIMBS),
                Register::Local(
                    index + NUM_CARRY_LIMBS + NUM_WITNESS_LIMBS,
                    NUM_WITNESS_LIMBS,
                ),
            ),
            Register::Next(index, length) => (
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
            c,
            d,
            output,
            witness,
            carry,
            witness_low,
            witness_high,
        }
    }

    #[inline]
    pub fn assign_row<T: Copy>(&self, trace_rows: &mut [Vec<T>], row: &mut [T], row_index: usize) {
        self.a.assign(trace_rows, &mut row[0..N_LIMBS], row_index);
        self.b
            .assign(trace_rows, &mut row[N_LIMBS..2 * N_LIMBS], row_index);
        self.c
            .assign(trace_rows, &mut row[2 * N_LIMBS..3 * N_LIMBS], row_index);
        self.d
            .assign(trace_rows, &mut row[3 * N_LIMBS..4 * N_LIMBS], row_index);
        self.output
            .assign(trace_rows, &mut row[4 * N_LIMBS..5 * N_LIMBS], row_index);
        self.witness
            .assign(trace_rows, &mut row[5 * N_LIMBS..], row_index)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    /// Returns a vector
    /// [Input[4 * N_LIMBS], output[N_LIMBS], carry[NUM_CARRY_LIMBS], Witness_low[NUM_WITNESS_LIMBS], Witness_high[NUM_WITNESS_LIMBS]]
    pub fn quad_trace(a: BigUint, b: BigUint, c: BigUint, d: BigUint) -> Vec<F> {
        let p = get_p();
        let result = (&a * &b + &c * &d) % &p;
        debug_assert!(result < p);
        let carry = (&a * &b + &c * &d - &result) / &p;
        debug_assert!(carry < 2u32 * &p);
        debug_assert_eq!(&carry * &p, &a * &b + &c * &d - &result);

        // make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(&a, 16, N_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(&b, 16, N_LIMBS);
        let p_c = Polynomial::<i64>::from_biguint_num(&c, 16, N_LIMBS);
        let p_d = Polynomial::<i64>::from_biguint_num(&d, 16, N_LIMBS);
        let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, N_LIMBS);

        let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
        let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, NUM_CARRY_LIMBS);

        // Compute the vanishing polynomial
        let vanishing_poly = &p_a * &p_b + &p_c * &p_d - &p_result - &p_carry * &p_p;
        debug_assert_eq!(vanishing_poly.degree(), NUM_WITNESS_LIMBS + 1);

        // Compute the witness
        let witness_shifted = extract_witness_and_shift(&vanishing_poly, WITNESS_OFFSET as u32);
        let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

        let mut row = Vec::with_capacity(NUM_QUAD_COLUMNS);

        // inputs
        row.extend(to_field_iter::<F>(&p_a));
        row.extend(to_field_iter::<F>(&p_b));
        row.extend(to_field_iter::<F>(&p_c));
        row.extend(to_field_iter::<F>(&p_d));

        // output
        row.extend(to_field_iter::<F>(&p_result));
        // carry and witness
        row.extend(to_field_iter::<F>(&p_carry));
        row.extend(witness_low);
        row.extend(witness_high);

        row
    }

    /// Quad generic constraints
    pub fn quad_packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        layout: QuadLayout,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // get all the data
        let a = layout.a.packed_entries_slice(&vars);
        let b = layout.b.packed_entries_slice(&vars);
        let c = layout.c.packed_entries_slice(&vars);
        let d = layout.d.packed_entries_slice(&vars);
        let output = layout.output.packed_entries_slice(&vars);

        let carry = layout.carry.packed_entries_slice(&vars);
        let witness_low = layout.witness_low.packed_entries_slice(&vars);
        let witness_high = layout.witness_high.packed_entries_slice(&vars);

        // Construct the expected vanishing polynmial
        let ab = PolynomialOps::mul(a, b);
        let cd = PolynomialOps::mul(c, d);
        let ab_plus_cd = PolynomialOps::add(&ab, &cd);
        let ab_plus_cd_minus_output = PolynomialOps::sub(&ab_plus_cd, output);
        let mul_times_carry = PolynomialOps::mul(&ab_plus_cd_minus_output, carry);
        let vanishing_poly = PolynomialOps::sub(&ab_plus_cd_minus_output, &mul_times_carry);

        // reconstruct witness
        let p_limbs = Polynomial::<FE>::from_iter(P_iter());

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

        debug_assert!(vanishing_poly.len() == witness_times_root.len());
        for i in 0..vanishing_poly.len() {
            yield_constr.constraint_transition(vanishing_poly[i] - witness_times_root[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::arithmetic::arithmetic_stark::{ArithmeticStark, EmulatedCircuitLayout};
    use crate::arithmetic::{ArithmeticLayout, ArithmeticOp};
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[test]
    fn test_quad_trace_generation() {
        let num_tests = 100;
        let p = get_p();
        const D: usize = 2;
        type F = <PoseidonGoldilocksConfig as GenericConfig<D>>::F;

        for _ in 0..num_tests {
            let a = rand::thread_rng().gen_biguint(256) % &p;
            let b = rand::thread_rng().gen_biguint(256) & &p;
            let c = rand::thread_rng().gen_biguint(256) & &p;
            let d = rand::thread_rng().gen_biguint(256) & &p;

            let _ =
                ArithmeticParser::<F, 4>::quad_trace(a.clone(), b.clone(), c.clone(), d.clone());
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct QuadLayoutCircuit;

    const LAYOUT: QuadLayout = QuadLayout::new(
        Register::Local(0, N_LIMBS),
        Register::Local(N_LIMBS, N_LIMBS),
        Register::Local(2 * N_LIMBS, N_LIMBS),
        Register::Local(3 * N_LIMBS, N_LIMBS),
        Register::Local(4 * N_LIMBS, N_LIMBS),
        Register::Local(5 * N_LIMBS, NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS),
    );

    /*   impl EmulatedCircuitLayout<1> for QuadLayoutCircuit {
        const PUBLIC_INPUTS: usize = 0;
        const ENTRY_COLUMN: usize = 0;
        const NUM_ARITHMETIC_COLUMNS: usize = NUM_QUAD_COLUMNS;
        const TABLE_INDEX: usize = NUM_QUAD_COLUMNS;

        const OPERATIONS: [ArithmeticLayout; 1] =
            [ArithmeticLayout::EdCurve(EdOpcodeLayout::Quad(LAYOUT))];
    }

    #[test]
    fn test_quad_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ArithmeticStark<QuadLayoutCircuit, 1, F, D>;

        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = get_p();

        let mut rng = rand::thread_rng();

        let mut quad_operations = Vec::new();

        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(255) % &p22519;
            let b = rng.gen_biguint(255) % &p22519;
            let c = rng.gen_biguint(255) % &p22519;
            let d = rng.gen_biguint(255) % &p22519;

            let operation = ArithmeticOp::EdCurveOp(EdOpcode::Quad(a, b, c, d));
            quad_operations.push((operation, 0));
        }

        let stark = S::new();

        let trace = stark.generate_trace(quad_operations);

        // Verify proof as a stark
        let proof =
            prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default()).unwrap();
        verify_stark_proof(stark, proof.clone(), &config).unwrap();
    }*/
}
