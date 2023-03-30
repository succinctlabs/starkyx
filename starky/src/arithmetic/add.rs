//! Implementation of modular addition as a STARK (prototype)
//!
//! The implementation based on a method used in Polygon starks
//!
//!
//!

use core::marker::PhantomData;

use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::util::transpose;

use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub const N_LIMBS: usize = 16;
pub const NUM_ARITH_COLUMNS: usize = 6 * N_LIMBS;
const RANGE_MAX: usize = 1usize << 16; // Range check strict upper bound

#[derive(Clone, Debug)]
pub struct ArithmeticOpStark<F, const D: usize> {
    program: Vec<(ArithmeticOp, ArithmeticMem)>,
    _marker: PhantomData<F>,
}

/// An experimental parser to generate Stark constaint code from commands
///
/// The output is writing to a "memory" passed to it.
#[derive(Debug, Clone, Copy)]
pub struct ArithmeticParser<F, const D: usize> {
    _marker: PhantomData<F>,
}

#[derive(Debug, Clone, Copy)]
pub enum Register {
    Local(usize, usize),
    Next(usize, usize),
}

impl Register {
    fn get_range(&self) -> (usize, usize) {
        match self {
            Register::Local(index, length) => (*index, *index + length),
            Register::Next(index, length) => (*index, *index + length),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ArithmeticOp {
    AddMod(BigUint, BigUint, BigUint),
    SubMod(BigUint, BigUint, BigUint),
    MulMod(BigUint, BigUint, BigUint),
}

#[derive(Debug, Clone, Copy)]
pub struct ArithmeticMem {
    input_1: Register,
    input_2: Register,
    output: Register,
    modulus: Register,
    carry: Register,
    witness: Register,
}

pub struct Trace<F> {
    trace_rows: Vec<Vec<F>>,
    current_row: usize,
}

impl<F: Field> Trace<F> {
    fn new(num_rows: usize, num_cols: usize) -> Self {
        let trace_rows = vec![vec![F::ZERO; num_cols]; num_rows];
        Self {
            trace_rows,
            current_row: 0,
        }
    }
    fn advance(&mut self) {
        self.current_row += 1;
    }

    fn alloc(&mut self, register: Register, value: &[F]) {
        match register {
            Register::Local(index, length) => {
                assert_eq!(length, value.len());
                for (i, v) in value.iter().enumerate() {
                    self.trace_rows[self.current_row][index + i] = *v;
                }
            }
            Register::Next(_, _) => unimplemented!("Next row allocation not implemented yet"),
        }
    }

    fn trace_cols(&self) -> Vec<PolynomialValues<F>> {
        let trace_cols = transpose(&self.trace_rows);
        trace_cols.into_iter().map(PolynomialValues::new).collect()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    fn op_trace(trace: &mut Trace<F>, operation: &ArithmeticOp, mem_alloc: ArithmeticMem) {
        match operation {
            ArithmeticOp::AddMod(a, b, m) => {
                Self::add_trace(trace, a, b, m, mem_alloc);
            }
            _ => unimplemented!("Operation not supported yet"),
        }
    }

    /// Converts two BigUint inputs into the correspinding rows of addition mod modulus
    ///
    /// a + b = c mod m
    ///
    /// Each element represented by a polynomial a(x), b(x), c(x), m(x) of 16 limbs of 16 bits each
    /// We will witness the relation
    ///  a(x) + b(x) - c(x) - carry(x) * m(x) - (x - 2^16) * s(x) == 0
    /// only a(x), b(x), c(x), m(x) should be range-checked.
    /// where carry = 0 or carry = 1
    /// the first row will contain a(x), b(x), m(x) and the second row will contain c(x), q(x), s(x)
    fn add_trace(
        trace: &mut Trace<F>,
        a: &BigUint,
        b: &BigUint,
        modulus: &BigUint,
        mem_alloc: ArithmeticMem,
    ) {
        // Calculate all results as BigUint
        let result = (a + b) % modulus;
        debug_assert!(&result < modulus);
        let carry = (a + b - &result) / modulus;
        debug_assert!(carry == BigUint::from(0u32) || carry == BigUint::from(1u32));

        // Make polynomial limbs
        let p_a = Polynomial::<F>::from_biguint(a, 16, N_LIMBS);
        let p_b = Polynomial::<F>::from_biguint(b, 16, N_LIMBS);
        let p_m = Polynomial::<F>::from_biguint(modulus, 16, N_LIMBS);
        let p_res = Polynomial::<F>::from_biguint(&result, 16, N_LIMBS);
        let p_c = Polynomial::<F>::from_biguint(&carry, 16, N_LIMBS);
        let carry_bit = p_c.as_slice()[0];

        // Make the witness polynomial
        let vanishing_poly = &p_a + &p_b - &p_res - &p_m * carry_bit;

        let limb = F::from_canonical_u32(2u32.pow(16));
        let witness_poly = vanishing_poly.root_quotient(limb);

        // Allocate the values to the trace
        trace.alloc(mem_alloc.input_1, p_a.as_slice());
        trace.alloc(mem_alloc.input_2, p_b.as_slice());
        trace.alloc(mem_alloc.output, p_res.as_slice());
        trace.alloc(mem_alloc.modulus, p_m.as_slice());
        trace.alloc(mem_alloc.carry, p_c.as_slice());
        trace.alloc(mem_alloc.witness, witness_poly.as_slice());
    }

    /// Make the stark constrains for the AddMod operation
    fn op_packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        operation: &ArithmeticOp,
        mem_alloc: ArithmeticMem,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match operation {
            ArithmeticOp::AddMod(_, _, _) => {
                Self::add_packed_generic_constraints(mem_alloc, vars, yield_constr);
            }
            _ => unimplemented!("Operation not supported yet"),
        }
    }

    fn add_packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        mem_alloc: ArithmeticMem,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // Get the range of the variables
        let (a_l, a_h) = mem_alloc.input_1.get_range();
        let (b_l, b_h) = mem_alloc.input_2.get_range();
        let (r_l, r_h) = mem_alloc.output.get_range();
        let (m_l, m_h) = mem_alloc.modulus.get_range();
        let (c_l, c_h) = mem_alloc.carry.get_range();
        let (w_l, w_h) = mem_alloc.witness.get_range();

        // Make polynomial limbs
        let a = &vars.local_values[a_l..a_h];
        let b = &vars.local_values[b_l..b_h];
        let m = &vars.local_values[m_l..m_h];
        let r = &vars.local_values[r_l..r_h];
        let c = &vars.local_values[c_l..c_h];
        let w = &vars.local_values[w_l..w_h];

        let limb: P = P::Scalar::from_canonical_u32(2u32.pow(16)).into();

        // Construct the vanishing polynomial
        let a_plus_b = PolynomialOps::add(a, b);
        let a_plus_b_minus_result = PolynomialOps::sub(&a_plus_b, r);
        let carry_times_mod = PolynomialOps::scalar_mul(m, &c[0]);
        let vanising_poly = PolynomialOps::sub(&a_plus_b_minus_result, &carry_times_mod);

        // Multiply by (x-2^16) and make the constraint
        let root_monomial: &[P] = &[-limb, P::from(P::Scalar::ONE)];
        let witness_times_root = PolynomialOps::mul(w, root_monomial);

        debug_assert!(vanising_poly.len() == witness_times_root.len());
        for i in 0..vanising_poly.len() {
            yield_constr.constraint_transition(vanising_poly[i] - witness_times_root[i]);
        }
    }

    fn op_ext_circuits<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        operation: &ArithmeticOp,
        mem_alloc: ArithmeticMem,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match operation {
            ArithmeticOp::AddMod(_, _, _) => {
                Self::op_add_ext_circuit(mem_alloc, builder, vars, yield_constr);
            }
            _ => unimplemented!("Operation not supported yet"),
        }
    }

    fn op_add_ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        mem_alloc: ArithmeticMem,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // Get the range of the variables
        let (a_l, a_h) = mem_alloc.input_1.get_range();
        let (b_l, b_h) = mem_alloc.input_2.get_range();
        let (r_l, r_h) = mem_alloc.output.get_range();
        let (m_l, m_h) = mem_alloc.modulus.get_range();
        let (c_l, c_h) = mem_alloc.carry.get_range();
        let (w_l, w_h) = mem_alloc.witness.get_range();

        // Make polynomial limbs
        let a = &vars.local_values[a_l..a_h];
        let b = &vars.local_values[b_l..b_h];
        let m = &vars.local_values[m_l..m_h];
        let r = &vars.local_values[r_l..r_h];
        let c = &vars.local_values[c_l..c_h];
        let w = &vars.local_values[w_l..w_h];

        // Construct the vanishing polynomial
        let a_plus_b = PolynomialGadget::add_extension(builder, a, b);
        let a_plus_b_minus_result = PolynomialGadget::sub_extension(builder, &a_plus_b, r);
        let carry_times_mod = PolynomialGadget::ext_scalar_mul_extension(builder, m, &c[0]);
        let vanising_poly =
            PolynomialGadget::sub_extension(builder, &a_plus_b_minus_result, &carry_times_mod);

        // Multiply by (x-2^16) and make the constraint
        let neg_limb = builder.constant_extension(-F::Extension::from_canonical_u32(2u32.pow(16)));
        let root_monomial = &[neg_limb, builder.constant_extension(F::Extension::ONE)];
        let witness_times_root = PolynomialGadget::mul_extension(builder, w, root_monomial);

        debug_assert!(vanising_poly.len() == witness_times_root.len());
        let constraint =
            PolynomialGadget::sub_extension(builder, &vanising_poly, &witness_times_root);
        for constr in constraint {
            yield_constr.constraint_transition(builder, constr);
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticOpStark<F, D> {
    fn gen_trace(&self) -> Vec<PolynomialValues<F>> {
        let num_rows = self.program.len();
        let num_columns = NUM_ARITH_COLUMNS;
        let mut trace = Trace::<F>::new(num_rows, num_columns);

        for (operation, mem_alloc) in self.program.iter() {
            ArithmeticParser::op_trace(&mut trace, operation, *mem_alloc);
            trace.advance();
        }
        trace.trace_cols()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ArithmeticOpStark<F, D> {
    const COLUMNS: usize = NUM_ARITH_COLUMNS;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mem_alloc = &self.program[0].1;
        ArithmeticParser::add_packed_generic_constraints(*mem_alloc, vars, yield_constr);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let mem_alloc = &self.program[0].1;
        ArithmeticParser::op_add_ext_circuit(*mem_alloc, builder, vars, yield_constr);
    }

    fn constraint_degree(&self) -> usize {
        2
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
    fn test_arithmetic_op_add() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ArithmeticOpStark<F, D>;

        let num_rows = 2u64.pow(13);
        let config = StarkConfig::standard_fast_config();

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        let mut additions = Vec::new();

        let input_1_index = 0;
        let input_2_index = N_LIMBS;
        let modulus_index = 2 * N_LIMBS;
        let output_index = 3 * N_LIMBS;
        let carry_index = 4 * N_LIMBS;
        let witness_index = 5 * N_LIMBS;

        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(255) % &p22519;
            let b = rng.gen_biguint(255) % &p22519;
            let p = p22519.clone();

            let operation = ArithmeticOp::AddMod(a.clone(), b.clone(), p.clone());
            let mem_alloc = ArithmeticMem {
                input_1: Register::Local(input_1_index, N_LIMBS),
                input_2: Register::Local(input_2_index, N_LIMBS),
                modulus: Register::Local(modulus_index, N_LIMBS),
                output: Register::Local(output_index, N_LIMBS),
                carry: Register::Local(carry_index, N_LIMBS),
                witness: Register::Local(witness_index, N_LIMBS - 1),
            };
            additions.push((operation, mem_alloc));
        }

        let stark = S {
            program: additions,
            _marker: PhantomData,
        };

        let trace = stark.gen_trace();

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
