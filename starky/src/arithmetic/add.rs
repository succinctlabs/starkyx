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
use crate::arithmetic::util::biguint_to_16_digits;
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
        trace.alloc(mem_alloc.input_1, &p_a.as_slice());
        trace.alloc(mem_alloc.input_2, &p_b.as_slice());
        trace.alloc(mem_alloc.output, &p_res.as_slice());
        trace.alloc(mem_alloc.modulus, &p_m.as_slice());
        trace.alloc(mem_alloc.carry, &p_c.as_slice());
        trace.alloc(mem_alloc.witness, &witness_poly.as_slice());
    }

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
        let witness_times_root = PolynomialOps::mul(&w, root_monomial);

        debug_assert!(vanising_poly.len() == witness_times_root.len());
        for i in 0..vanising_poly.len() {
            yield_constr.constraint_transition(vanising_poly[i] - witness_times_root[i]);
        }
    }

    fn op_circuits() {}
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
        for (operation, mem_alloc) in self.program.iter() {
            match operation {
                ArithmeticOp::AddMod(_, _, _) => {
                    ArithmeticParser::op_packed_generic_constraints(
                        operation,
                        *mem_alloc,
                        vars,
                        yield_constr,
                    );
                }
                _ => unimplemented!("Operation not supported yet"),
            }
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        2
    }
}

// Old Add code

#[derive(Copy, Clone)]
pub struct AddModStark<F, const D: usize> {
    _marker: PhantomData<F>,
}

type AdditionTuple = (BigUint, BigUint, BigUint);

impl<F: RichField + Extendable<D>, const D: usize> AddModStark<F, D> {
    /// Generate trace for addition stark
    fn generate_trace(&self, additions: Vec<AdditionTuple>) -> Vec<PolynomialValues<F>> {
        let max_rows = core::cmp::max(2 * additions.len(), RANGE_MAX); // note : range_max not needed yet
        let mut trace_rows: Vec<Vec<F>> = Vec::with_capacity(max_rows);

        for (a, b, m) in additions {
            let row = ArithmeticParser::<F, D>::add_to_rows(a, b, m);
            trace_rows.push(row);
        }

        let trace_cols = transpose(&trace_rows);

        trace_cols.into_iter().map(PolynomialValues::new).collect()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> ArithmeticParser<F, D> {
    /// Converts two BigUint inputs into the correspinding rows of addition mod modulus
    ///
    /// a + b = c mod m
    ///
    /// Each element represented by a polynomial a(x), b(x), c(x), m(x) of 16 limbs of 16 bits each
    /// We will witness the relation
    ///  a(x) + b(x) - c(x) - carry * m(x) - (x - 2^16) * s(x) == 0
    /// only a(x), b(x), c(x), m(x) should be range-checked.
    /// where carry = 0 or carry = 1
    /// the first row will contain a(x), b(x), m(x) and the second row will contain c(x), q(x), s(x)
    pub fn add_to_rows(input_0: BigUint, input_1: BigUint, modulus: BigUint) -> Vec<F> {
        let result = (&input_0 + &input_1) % &modulus;
        debug_assert!(result < modulus);
        let carry = (&input_0 + &input_1 - &result) / &modulus;
        debug_assert!(carry == BigUint::from(0u32) || carry == BigUint::from(1u32));

        let carry_digits = biguint_to_16_digits(&carry, N_LIMBS);

        let mut row = vec![F::ZERO; NUM_ARITH_COLUMNS];

        let input_0_digits = biguint_to_16_digits(&input_0, N_LIMBS);
        let input_1_digits = biguint_to_16_digits(&input_1, N_LIMBS);
        let result_digits = biguint_to_16_digits(&result, N_LIMBS);
        let modulus_digits = biguint_to_16_digits(&modulus, N_LIMBS);

        let carry_mod = &carry * &modulus;
        let carry_mod_digits = biguint_to_16_digits(&carry_mod, N_LIMBS);
        for i in 0..N_LIMBS {
            assert!(carry_mod_digits[i] == F::ZERO || carry_mod_digits[i] == modulus_digits[i]);
        }

        // constr_poly is the array of coefficients of the polynomial
        //
        // a(x) +  b(x) - c(x) - carry*m(x) = const(x)
        // note that we don't care about the coefficients of constr(x) at all, just that it will have a root.
        let consr_polynomial: Vec<F> = input_0_digits
            .iter()
            .zip(input_1_digits.iter())
            .zip(result_digits.iter())
            .zip(carry_mod_digits.iter())
            .map(|(((a, b), r), mc)| *a + *b - *r - *mc)
            .collect();

        assert_eq!(consr_polynomial.len(), N_LIMBS);
        // By assumption β := 2^16 is a root of `a`, i.e. (x - β) divides
        // `a`; if we write
        //
        //    a(x) = \sum_{i=0}^{N-1} a[i] x^i
        //         = (x - β) \sum_{i=0}^{N-2} q[i] x^i
        //
        // then by comparing coefficients it is easy to see that
        //
        //   q[0] = -a[0] / β  and  q[i] = (q[i-1] - a[i]) / β
        //
        //  NOTE : Doing divisions in F::Goldilocks probably not the most efficient
        let mut aux_digits = Vec::new();
        let pow_2 = F::from_canonical_u32(65536u32);
        aux_digits.push(-consr_polynomial[0] / F::from_canonical_u32(65536u32));

        for deg in 1..N_LIMBS - 1 {
            let temp1 = aux_digits[deg - 1];
            let digit = temp1 - consr_polynomial[deg];
            let quot = digit / F::from_canonical_u32(65536u32);
            aux_digits.push(quot);
        }
        aux_digits.push(F::ZERO);

        // Add inputs and modulus as values in first row
        for i in 0..N_LIMBS {
            row[i] = input_0_digits[i];
            row[i + N_LIMBS] = input_1_digits[i];
            row[i + 2 * N_LIMBS] = modulus_digits[i];
        }

        // Add result, quotient and aux polynomial as values in second row
        for i in 0..N_LIMBS {
            row[i + 3 * N_LIMBS] = result_digits[i];
            row[i + N_LIMBS + 3 * N_LIMBS] = carry_digits[i];
            row[i + 2 * N_LIMBS + 3 * N_LIMBS] = aux_digits[i];
        }

        // Check consr reconstruction
        // Calculates aux(x)*(x-2^16)
        let mut consr_sanity = vec![];

        consr_sanity.push(-row[2 * N_LIMBS + 3 * N_LIMBS].mul(pow_2));
        for i in 1..N_LIMBS - 1 {
            let val = -row[i + 2 * N_LIMBS + 3 * N_LIMBS].mul(pow_2)
                + row[i - 1 + 2 * N_LIMBS + 3 * N_LIMBS];
            consr_sanity.push(val)
        }
        assert_eq!(row[N_LIMBS - 1 + 2 * N_LIMBS + 3 * N_LIMBS], F::ZERO);
        consr_sanity.push(row[N_LIMBS - 2 + 2 * N_LIMBS + 3 * N_LIMBS]);
        assert_eq!(consr_sanity.len(), N_LIMBS);

        for i in 0..N_LIMBS {
            assert_eq!(consr_sanity[i], consr_polynomial[i]);
        }

        let mut sum_minus_carry_sanity = vec![];
        for i in 0..N_LIMBS {
            sum_minus_carry_sanity.push(
                row[i] + row[i + N_LIMBS]
                    - row[i + 3 * N_LIMBS]
                    - row[N_LIMBS + 3 * N_LIMBS] * row[i + 2 * N_LIMBS],
            );
        }

        for i in 0..N_LIMBS {
            assert_eq!(sum_minus_carry_sanity[i], consr_polynomial[i]);
        }

        row
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for AddModStark<F, D> {
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
        // a(x) + b(x) - c(x) - carry * m(x) - (x - β) * s(x) == 0
        // the first row = (a(x), b(x), m(x)) and the second ro = (c(x), carry(x), s(x))
        let mut sum_minus_carry = vec![];
        for i in 0..N_LIMBS {
            sum_minus_carry.push(
                vars.local_values[i] + vars.local_values[i + N_LIMBS]
                    - vars.local_values[i + 3 * N_LIMBS]
                    - vars.local_values[N_LIMBS + 3 * N_LIMBS] * vars.local_values[i + 2 * N_LIMBS],
            );
        }

        let mut consr_poly = vec![];
        let pow_2 = P::Scalar::from_canonical_u32(65536u32);

        consr_poly.push(-vars.local_values[2 * N_LIMBS + 3 * N_LIMBS].mul(pow_2));
        for i in 1..N_LIMBS - 1 {
            let val = -vars.local_values[i + 2 * N_LIMBS + 3 * N_LIMBS].mul(pow_2)
                + vars.local_values[i - 1 + 2 * N_LIMBS + 3 * N_LIMBS];
            consr_poly.push(val)
        }
        consr_poly.push(vars.local_values[N_LIMBS - 2 + 2 * N_LIMBS + 3 * N_LIMBS]);
        for i in 0..N_LIMBS {
            yield_constr.constraint_transition(sum_minus_carry[i] - consr_poly[i]);
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        // a(x) + b(x) - c(x) - carry * m(x) - (x - β) * s(x) == 0
        // the first row = (a(x), b(x), m(x)) and the second ro = (c(x), carry(x), s(x))
        let mut sum_minus_carry = vec![];
        for i in 0..N_LIMBS {
            let sum = builder.add_extension(vars.local_values[i], vars.local_values[i + N_LIMBS]);
            let sum_minus_res = builder.sub_extension(sum, vars.local_values[i + 3 * N_LIMBS]);
            let carry_mod = builder.mul_extension(
                vars.local_values[N_LIMBS + 3 * N_LIMBS],
                vars.local_values[i + 2 * N_LIMBS],
            );
            sum_minus_carry.push(builder.sub_extension(sum_minus_res, carry_mod));
        }

        let pow_2 = F::from_canonical_u32(2u32.pow(16));

        let mut consr_poly = vec![];
        let const_term_neg_pow_2 =
            builder.mul_const_extension(pow_2, vars.local_values[2 * N_LIMBS + 3 * N_LIMBS]);
        let zero = builder.constant_extension(<F as Extendable<D>>::Extension::from(F::ZERO));
        let const_term = builder.sub_extension(zero, const_term_neg_pow_2);

        consr_poly.push(const_term);
        for i in 1..N_LIMBS - 1 {
            let pow_2_aux = builder
                .mul_const_extension(pow_2, vars.local_values[i + 2 * N_LIMBS + 3 * N_LIMBS]);
            let val = builder.sub_extension(
                vars.local_values[i - 1 + 2 * N_LIMBS + 3 * N_LIMBS],
                pow_2_aux,
            );
            consr_poly.push(val);
        }
        consr_poly.push(vars.local_values[N_LIMBS - 2 + 2 * N_LIMBS + 3 * N_LIMBS]);

        for i in 0..N_LIMBS {
            let constraint = builder.sub_extension(sum_minus_carry[i], consr_poly[i]);
            yield_constr.constraint_transition(builder, constraint);
        }
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
    fn test_add_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = AddModStark<F, D>;

        let num_rows = 32;

        let config = StarkConfig::standard_fast_config();

        let stark = S {
            _marker: PhantomData,
        };

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        let mut additions = Vec::new();
        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(255) % &p22519;
            let b = rng.gen_biguint(255) % &p22519;
            let p = p22519.clone();
            additions.push((a, b, p));
        }

        let trace = stark.generate_trace(additions);

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

        //timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }

    #[test]
    fn test_arihtmetic_op_add() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ArithmeticOpStark<F, D>;

        let num_rows = 32;
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
        verify_stark_proof(stark, proof.clone(), &config).unwrap();
    }
}
