use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::*;
use crate::arithmetic::builder::ChipBuilder;
use crate::arithmetic::chip::ChipParameters;
use crate::arithmetic::field::modulus_field_iter;
use crate::arithmetic::instruction::Instruction;
use crate::arithmetic::polynomial::{Polynomial, PolynomialGadget, PolynomialOps};
use crate::arithmetic::register::{Array, MemorySlice, RegisterSerializable, U16Register};
use crate::arithmetic::trace::TraceHandle;
use crate::arithmetic::utils::{extract_witness_and_shift, split_digits, to_field_iter};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub struct Den<P: FieldParameters> {
    a: FieldRegister<P>,
    b: FieldRegister<P>,
    sign: bool,
    result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: Array<U16Register>,
    witness_high: Array<U16Register>,
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
    pub fn ed_den<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
        sign: bool,
        result: &FieldRegister<P>,
    ) -> Result<Den<P>>
    where
        L::Instruction: From<Den<P>>,
    {
        let carry = self.alloc::<FieldRegister<P>>().unwrap();
        let witness_low = self
            .alloc_array::<U16Register>(P::NB_WITNESS_LIMBS)
            .unwrap();
        let witness_high = self
            .alloc_array::<U16Register>(P::NB_WITNESS_LIMBS)
            .unwrap();
        let instr = Den {
            a: *a,
            b: *b,
            sign,
            result: *result,
            carry,
            witness_low,
            witness_high,
        };
        self.insert_instruction(instr.into())?;
        Ok(instr)
    }
}

impl<F: RichField + Extendable<D>, const D: usize, P: FieldParameters> Instruction<F, D>
    for Den<P>
{
    fn layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.result.register(),
            *self.carry.register(),
            *self.witness_low.register(),
            *self.witness_high.register(),
        ]
    }

    fn packed_generic_constraints<
        FE,
        PF,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, PF, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<PF>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        PF: PackedField<Scalar = FE>,
    {
        // get all the data
        let a = self.a.register().packed_entries_slice(&vars);
        let b = self.b.register().packed_entries_slice(&vars);
        let result = self.result.register().packed_entries_slice(&vars);

        let carry = self.carry.register().packed_entries_slice(&vars);
        let witness_low = self.witness_low.register().packed_entries_slice(&vars);
        let witness_high = self.witness_high.register().packed_entries_slice(&vars);

        let p_limbs = Polynomial::<FE>::from_iter(modulus_field_iter::<FE, P>());

        // equation_lhs = if sign { b * result + result} else { b * result + a}
        let equation_lhs = if self.sign {
            PolynomialOps::add(&PolynomialOps::mul(b, result), result)
        } else {
            PolynomialOps::add(&PolynomialOps::mul(b, result), a)
        };
        // let equation_rhs = if sign { a } else { result };
        let equation_rhs = if self.sign { a } else { result };
        let lhs_minus_rhs = PolynomialOps::sub(&equation_lhs, equation_rhs);
        let mul_times_carry = PolynomialOps::scalar_poly_mul(carry, p_limbs.as_slice());
        let vanishing_poly = PolynomialOps::sub(&lhs_minus_rhs, &mul_times_carry);

        // reconstruct witness

        let limb = FE::from_canonical_u32(LIMB);

        // Reconstruct and shift back the witness polynomial
        let w_shifted = witness_low
            .iter()
            .zip(witness_high.iter())
            .map(|(x, y)| *x + (*y * limb));

        let offset = FE::from_canonical_u32(P::WITNESS_OFFSET as u32);
        let w = w_shifted.map(|x| x - offset).collect::<Vec<PF>>();

        // Multiply by (x-2^16) and make the constraint
        let root_monomial: &[PF] = &[PF::from(-limb), PF::from(PF::Scalar::ONE)];
        let witness_times_root = PolynomialOps::mul(&w, root_monomial);

        //debug_assert!(vanishing_poly.len() == witness_times_root.len());
        for i in 0..vanishing_poly.len() {
            yield_constr.constraint(vanishing_poly[i] - witness_times_root[i]);
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

        let carry = self.carry.register().evaluation_targets(&vars);
        let witness_low = self.witness_low.register().evaluation_targets(&vars);
        let witness_high = self.witness_high.register().evaluation_targets(&vars);

        let p_limbs = PolynomialGadget::constant_extension(
            builder,
            &modulus_field_iter::<F::Extension, P>().collect::<Vec<_>>(),
        );

        // equation_lhs = if sign { b * result + result} else { b * result + a}
        let b_times_res = PolynomialGadget::mul_extension(builder, b, result);
        let equation_lhs = if self.sign {
            PolynomialGadget::add_extension(builder, &b_times_res, result)
        } else {
            PolynomialGadget::add_extension(builder, &b_times_res, a)
        };
        let equation_rhs = if self.sign { a } else { result };
        let lhs_minus_rhs = PolynomialGadget::sub_extension(builder, &equation_lhs, equation_rhs);

        // Construct the expected vanishing polynmial
        // let res_z = PolynomialGadget::mul_extension(builder, result, &z);
        // let res_z_minus_a = PolynomialGadget::sub_extension(builder, &res_z, a);
        let mul_times_carry = PolynomialGadget::mul_extension(builder, carry, &p_limbs[..]);
        let vanishing_poly =
            PolynomialGadget::sub_extension(builder, &lhs_minus_rhs, &mul_times_carry);

        // reconstruct witness

        // Reconstruct and shift back the witness polynomial
        let limb_const = F::Extension::from_canonical_u32(2u32.pow(16));
        let limb = builder.constant_extension(limb_const);
        let w_high_times_limb =
            PolynomialGadget::ext_scalar_mul_extension(builder, witness_high, &limb);
        let w_shifted = PolynomialGadget::add_extension(builder, witness_low, &w_high_times_limb);
        let offset =
            builder.constant_extension(F::Extension::from_canonical_u32(P::WITNESS_OFFSET as u32));
        let w = PolynomialGadget::sub_constant_extension(builder, &w_shifted, &offset);

        // Multiply by (x-2^16) and make the constraint
        let neg_limb = builder.constant_extension(-limb_const);
        let root_monomial = &[neg_limb, builder.constant_extension(F::Extension::ONE)];
        let witness_times_root =
            PolynomialGadget::mul_extension(builder, w.as_slice(), root_monomial);

        let constraint =
            PolynomialGadget::sub_extension(builder, &vanishing_poly, &witness_times_root);

        for constr in constraint {
            yield_constr.constraint(builder, constr);
        }
    }
}

impl<P: FieldParameters> Den<P> {
    /// Trace row for fp_mul operation
    ///
    /// Returns a vector
    /// [Input[2 * N_LIMBS], output[N_LIMBS], carry[NUM_CARRY_LIMBS], Witness_low[NUM_WITNESS_LIMBS], Witness_high[NUM_WITNESS_LIMBS]]
    pub fn trace_row<F: RichField + Extendable<D>, const D: usize>(
        a: &BigUint,
        b: &BigUint,
        sign: bool,
    ) -> (Vec<F>, BigUint) {
        let p = P::modulus_biguint();

        let minus_b_int = &p - b;
        let b_signed = if sign { b } else { &minus_b_int };

        let denominator = (b_signed + 1u32) % &p;
        let den_inv = denominator.modpow(&(&p - 2u32), &p);
        debug_assert_eq!(&den_inv * &denominator % &p, BigUint::from(1u32));

        let result = (a * &den_inv) % &p;
        debug_assert!(result < p);

        let equation_lhs = if sign {
            b * &result + &result
        } else {
            b * &result + a
        };
        let equation_rhs = if sign { a.clone() } else { result.clone() };

        let carry = (&equation_lhs - &equation_rhs) / &p;
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, &equation_lhs - &equation_rhs);

        // make polynomial limbs
        let p_a = Polynomial::<i64>::from_biguint_num(a, 16, P::NB_LIMBS);
        let p_b = Polynomial::<i64>::from_biguint_num(b, 16, P::NB_LIMBS);
        let p_p = Polynomial::<i64>::from_biguint_num(&p, 16, P::NB_LIMBS);

        let p_result = Polynomial::<i64>::from_biguint_num(&result, 16, P::NB_LIMBS);
        let p_carry = Polynomial::<i64>::from_biguint_num(&carry, 16, P::NB_LIMBS);

        // Compute the vanishing polynomial
        let vanishing_poly = if sign {
            &p_b * &p_result + &p_result - &p_a - &p_carry * &p_p
        } else {
            &p_b * &p_result + &p_a - &p_result - &p_carry * &p_p
        };
        debug_assert_eq!(vanishing_poly.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness
        let witness_shifted = extract_witness_and_shift(&vanishing_poly, P::WITNESS_OFFSET as u32);
        let (witness_low, witness_high) = split_digits::<F>(&witness_shifted);

        let mut row = Vec::new();
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
    pub fn write_ed_den<P: FieldParameters>(
        &self,
        row_index: usize,
        a_int: &BigUint,
        b_int: &BigUint,
        sign: bool,
        instruction: Den<P>,
    ) -> Result<BigUint> {
        let (row, result) = Den::<P>::trace_row::<F, D>(a_int, b_int, sign);
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
    //use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::arithmetic::builder::ChipBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
    use crate::arithmetic::field::{Fp25519, Fp25519Param};
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    struct DenTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for DenTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 124;
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = Den<Fp25519Param>;
    }

    #[test]
    fn test_ed_den() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<DenTest, F, D>;

        // build the stark
        let mut builder = ChipBuilder::<DenTest, F, D>::new();

        let a = builder.alloc::<Fp>().unwrap();
        let b = builder.alloc::<Fp>().unwrap();
        let sign = false;
        let result = builder.alloc::<Fp>().unwrap();

        let den_ins = builder.ed_den(&a, &b, sign, &result).unwrap();
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
            //let handle = handle.clone();
            //rayon::spawn(move || {
            handle.write_field(i as usize, &a_int, a).unwrap();
            handle.write_field(i as usize, &b_int, b).unwrap();
            let res = handle
                .write_ed_den(i as usize, &a_int, &b_int, sign, den_ins)
                .unwrap();
            //});
            if sign {
                assert_eq!(res, (a_int * (1u32 + b_int).modpow(&(&p - 2u32), &p)) % &p);
            } else {
                assert_eq!(
                    res,
                    (a_int * (1u32 + &p - b_int).modpow(&(&p - 2u32), &p)) % &p
                );
            }
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
