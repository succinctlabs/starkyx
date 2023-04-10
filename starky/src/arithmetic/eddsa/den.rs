use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::*;
use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::arithmetic::util::{extract_witness_and_shift, split_digits, to_field_iter};
use crate::arithmetic::{ArithmeticParser, Register};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub const NUM_CARRY_LIMBS: usize = N_LIMBS;
pub const NUM_WITNESS_LIMBS: usize = 2 * N_LIMBS - 2;
pub const NUM_DEN_COLUMNS: usize = 3 * N_LIMBS + NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS;
pub const TOTAL_WITNESS_COLUMNS: usize = NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS;

#[derive(Debug, Clone, Copy)]
pub struct DenLayout {
    a: Register,
    b: Register,
    sign: bool,
    output: Register,
    witness: Register,
    carry: Register,
    witness_low: Register,
    witness_high: Register,
}

impl DenLayout {
    #[inline]
    pub const fn new(
        a: Register,
        b: Register,
        sign: bool,
        output: Register,
        witness: Register,
    ) -> Self {
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
            sign,
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
    pub fn den_trace(a: BigUint, b: BigUint, sign: bool) -> (Vec<F>, BigUint) {
        let p = get_p();

        let b_signed = if sign { &b + 0u32 } else { &p - &b };

        let z = (&b_signed + 1u32) % &p;
        let z_inverse = z.modpow(&(&p - 2u32), &p);
        debug_assert_eq!(&z_inverse * &z % &p, BigUint::from(1u32));

        let result = (&a * &z_inverse) % &p;
        debug_assert!(result < p);
        let carry = (&result * &z - &a) / &p;
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, &z * &result - &a);

        // make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(&a, 16, N_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(&b, 16, N_LIMBS);
        let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, N_LIMBS);

        let p_b_sign = if sign { p_b } else { &p_p - &p_b };
        let p_z = &p_b_sign + 1;

        let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, N_LIMBS);
        let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, NUM_CARRY_LIMBS);

        // Compute the vanishing polynomial
        let vanishing_poly = &p_z * &p_result - &p_a - &p_carry * &p_p;
        debug_assert_eq!(vanishing_poly.degree(), NUM_WITNESS_LIMBS);

        // Compute the witness
        let witness_shifted = extract_witness_and_shift(&vanishing_poly, WITNESS_OFFSET as u32);
        let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

        let mut row = Vec::with_capacity(NUM_DEN_COLUMNS);

        // output
        row.extend(to_field_iter::<F>(&p_result));
        // carry and witness
        row.extend(to_field_iter::<F>(&p_carry));
        row.extend(witness_low);
        row.extend(witness_high);

        (row, result)
    }

    /// Quad generic constraints
    pub fn den_packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        layout: DenLayout,
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

        let p_limbs = Polynomial::<FE>::from_iter(P_iter());
        let p_p = Polynomial::<P>::from_polynomial(p_limbs.clone());

        // z = sign*b + 1
        let minus_b = PolynomialOps::sub(p_p.as_slice(), b);
        let mut z = if layout.sign { b.to_vec() } else { minus_b };
        z[0] += P::from(FE::ONE);

        let res_z = PolynomialOps::mul(output, &z);
        let res_z_minus_a = PolynomialOps::sub(&res_z, a);
        let mul_times_carry = PolynomialOps::scalar_poly_mul(carry, p_limbs.as_slice());
        let vanishing_poly = PolynomialOps::sub(&res_z_minus_a, &mul_times_carry);

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

    pub fn den_ext_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        layout: DenLayout,
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

        let p_limbs =
            PolynomialGadget::constant_extension(builder, &P_iter().collect::<Vec<_>>()[..]);

        let minus_b = PolynomialGadget::sub_extension(builder, &p_limbs, b);
        let mut z = builder.add_virtual_extension_targets(N_LIMBS);
        let one = builder.constant_extension(F::Extension::ONE);

        if layout.sign {
            z[0] = builder.add_extension(b[0], one);
            z[1..N_LIMBS].copy_from_slice(&b[1..N_LIMBS]);
            //for i in 1..N_LIMBS {
            //    z[i] = b[i];
            //}
        } else {
            z[0] = builder.add_extension(minus_b[0], one);
            z[1..N_LIMBS].copy_from_slice(&minus_b[1..N_LIMBS]);
            //for i in 1..N_LIMBS {
            //    z[i] = minus_b[i];
            //}
        }

        // Construct the expected vanishing polynmial
        let res_z = PolynomialGadget::mul_extension(builder, output, &z);
        let res_z_minus_a = PolynomialGadget::sub_extension(builder, &res_z, a);
        let mul_times_carry = PolynomialGadget::mul_extension(builder, carry, &p_limbs[..]);
        let vanishing_poly =
            PolynomialGadget::sub_extension(builder, &res_z_minus_a, &mul_times_carry);

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

    use super::*;
    use crate::arithmetic::arithmetic_stark::ArithmeticStark;
    use crate::arithmetic::chip::EmulatedCircuitLayout;
    use crate::arithmetic::InstructionT;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[test]
    fn test_denn_trace_generation() {
        let num_tests = 1 << 5;
        let p = get_p();
        const D: usize = 2;
        type F = <PoseidonGoldilocksConfig as GenericConfig<D>>::F;

        for _ in 0..num_tests {
            let a = rand::thread_rng().gen_biguint(256) % &p;
            let b = rand::thread_rng().gen_biguint(256) & &p;

            let _ = ArithmeticParser::<F, 4>::den_trace(a.clone(), b.clone(), false);
            let _ = ArithmeticParser::<F, 4>::den_trace(a.clone(), b.clone(), true);
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct DenLayoutCircuit;

    const LAYOUT: DenLayout = DenLayout::new(
        Register::Local(0, N_LIMBS),
        Register::Local(N_LIMBS, N_LIMBS),
        true,
        Register::Local(2 * N_LIMBS, N_LIMBS),
        Register::Local(3 * N_LIMBS, NUM_CARRY_LIMBS + 2 * NUM_WITNESS_LIMBS),
    );

    const INLAYOUT: WriteInputLayout = WriteInputLayout::new(Register::Local(0, 2 * N_LIMBS));

    impl<F: RichField + Extendable<D>, const D: usize> EmulatedCircuitLayout<F, D, 2>
        for DenLayoutCircuit
    {
        const PUBLIC_INPUTS: usize = 0;
        const ENTRY_COLUMN: usize = 0;
        const NUM_ARITHMETIC_COLUMNS: usize = NUM_DEN_COLUMNS;
        const TABLE_INDEX: usize = NUM_DEN_COLUMNS;

        type Layouts = EpOpcodewithInputLayout;

        const OPERATIONS: [EpOpcodewithInputLayout; 2] = [
            EpOpcodewithInputLayout::Ep(EdOpcodeLayout::DEN(LAYOUT)),
            EpOpcodewithInputLayout::Input(INLAYOUT),
        ];
    }

    #[derive(Debug, Clone)]
    pub struct DenlInstruction {
        pub a: BigUint,
        pub b: BigUint,
    }

    impl DenlInstruction {
        pub fn new(a: BigUint, b: BigUint) -> Self {
            Self { a, b }
        }
    }

    impl<F: RichField + Extendable<D>, const D: usize> InstructionT<DenLayoutCircuit, F, D, 2>
        for DenlInstruction
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
                let sign = match <DenLayoutCircuit as EmulatedCircuitLayout<F, D, 2>>::OPERATIONS[0]
                {
                    EpOpcodewithInputLayout::Ep(EdOpcodeLayout::DEN(layout)) => layout.sign,
                    _ => unreachable!(),
                };
                let operation = EdOpcode::DEN(self.a, self.b, sign);
                let (trace_row, _) = operation.generate_trace_row();
                tx.send((pc, 0, trace_row)).unwrap();
            });
        }
    }

    #[test]
    fn test_arithmetic_stark_den() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ArithmeticStark<DenLayoutCircuit, 2, F, D>;

        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = get_p();

        let mut rng = rand::thread_rng();

        let mut quad_operations = Vec::new();

        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(256) % &p22519;
            let b = rng.gen_biguint(256) % &p22519;

            quad_operations.push(DenlInstruction::new(a, b));
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
}
