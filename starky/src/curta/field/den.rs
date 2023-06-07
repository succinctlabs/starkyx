//! Implements non-native field "den" as an "instruction". This is used in ed25519 curve operations.
//!
//! To understand the implementation, it may be useful to refer to `mod.rs`.

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::constraint::{
    eval_field_operation, ext_circuit_field_operation, packed_generic_field_operation,
};
use super::*;
use crate::curta::air::parser::AirParser;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::field::modulus_field_iter;
use crate::curta::instruction::Instruction;
use crate::curta::parameters::FieldParameters;
use crate::curta::polynomial::parser::PolynomialParser;
use crate::curta::polynomial::{
    to_u16_le_limbs_polynomial, Polynomial, PolynomialGadget, PolynomialOps,
};
use crate::curta::register::{
    ArrayRegister, MemorySlice, Register, RegisterSerializable, U16Register,
};
use crate::curta::trace::writer::TraceWriter;
use crate::curta::utils::{compute_root_quotient_and_shift, split_u32_limbs_to_u16_limbs};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone, Copy)]
pub struct FpDenInstruction<P: FieldParameters> {
    a: FieldRegister<P>,
    b: FieldRegister<P>,
    sign: bool,
    pub result: FieldRegister<P>,
    carry: FieldRegister<P>,
    witness_low: ArrayRegister<U16Register>,
    witness_high: ArrayRegister<U16Register>,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub fn fp_den<P: FieldParameters>(
        &mut self,
        a: &FieldRegister<P>,
        b: &FieldRegister<P>,
        sign: bool,
    ) -> FpDenInstruction<P>
    where
        L::Instruction: From<FpDenInstruction<P>>,
    {
        let result = self.alloc::<FieldRegister<P>>();
        let carry = self.alloc::<FieldRegister<P>>();
        let witness_low = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let witness_high = self.alloc_array::<U16Register>(P::NB_WITNESS_LIMBS);
        let instr = FpDenInstruction {
            a: *a,
            b: *b,
            sign,
            result,
            carry,
            witness_low,
            witness_high,
        };
        self.constrain_instruction(instr.into()).unwrap();
        instr
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    pub fn write_fp_den<P: FieldParameters>(
        &self,
        row_index: usize,
        a: &BigUint,
        b: &BigUint,
        sign: bool,
        instruction: FpDenInstruction<P>,
    ) -> Result<BigUint> {
        let p = P::modulus();
        let minus_b_int = &p - b;
        let b_signed = if sign { b } else { &minus_b_int };
        let denominator = (b_signed + 1u32) % &p;
        let den_inv = denominator.modpow(&(&p - 2u32), &p);
        let result = (a * &den_inv) % &p;
        debug_assert_eq!(&den_inv * &denominator % &p, BigUint::from(1u32));
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

        // Make little endian polynomial limbs.
        let p_a = to_u16_le_limbs_polynomial::<F, P>(a);
        let p_b = to_u16_le_limbs_polynomial::<F, P>(b);
        let p_p = to_u16_le_limbs_polynomial::<F, P>(&p);
        let p_result = to_u16_le_limbs_polynomial::<F, P>(&result);
        let p_carry = to_u16_le_limbs_polynomial::<F, P>(&carry);

        // Compute the vanishing polynomial.
        let vanishing_poly = if sign {
            &p_b * &p_result + &p_result - &p_a - &p_carry * &p_p
        } else {
            &p_b * &p_result + &p_a - &p_result - &p_carry * &p_p
        };
        debug_assert_eq!(vanishing_poly.degree(), P::NB_WITNESS_LIMBS);

        // Compute the witness.
        let p_witness_shifted = compute_root_quotient_and_shift(&vanishing_poly, P::WITNESS_OFFSET);
        let (p_witness_low, p_witness_high) = split_u32_limbs_to_u16_limbs::<F>(&p_witness_shifted);

        // Row must match layout of instruction.
        self.write_to_layout(
            row_index,
            instruction,
            vec![
                p_result.coefficients,
                p_carry.coefficients,
                p_witness_low,
                p_witness_high,
            ],
        )?;
        Ok(result)
    }
}

impl<F: RichField + Extendable<D>, const D: usize, P: FieldParameters> Instruction<F, D>
    for FpDenInstruction<P>
{
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![
            *self.result.register(),
            *self.carry.register(),
            *self.witness_low.register(),
            *self.witness_high.register(),
        ]
    }

    fn packed_generic<FE, PF, const D2: usize, const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        vars: StarkEvaluationVars<FE, PF, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<PF>
    where
        FE: FieldExtension<D2, BaseField = F>,
        PF: PackedField<Scalar = FE>,
    {
        // Get the packed entries.
        let p_a = self.a.register().packed_generic_vars(vars);
        let p_b = self.b.register().packed_generic_vars(vars);
        let p_result = self.result.register().packed_generic_vars(vars);
        let p_carry = self.carry.register().packed_generic_vars(vars);
        let p_witness_low = self.witness_low.register().packed_generic_vars(vars);
        let p_witness_high = self.witness_high.register().packed_generic_vars(vars);

        // Compute the vanishing polynomial:
        //      lhs(x) = sign * (b(x) * result(x) + result(x)) + (1 - sign) * (b(x) * result(x) + a(x))
        //      rhs(x) = sign * a(x) + (1 - sign) * result(x)
        //      lhs(x) - rhs(x) - carry(x) * p(x)
        let p_equation_lhs = if self.sign {
            PolynomialOps::add(&PolynomialOps::mul(p_b, p_result), p_result)
        } else {
            PolynomialOps::add(&PolynomialOps::mul(p_b, p_result), p_a)
        };
        let p_equation_rhs = if self.sign { p_a } else { p_result };
        let p_lhs_minus_rhs = PolynomialOps::sub(&p_equation_lhs, p_equation_rhs);
        let p_modulus = Polynomial::<FE>::from_iter(modulus_field_iter::<FE, P>());
        let p_carry_mul_modulus = PolynomialOps::scalar_poly_mul(p_carry, p_modulus.as_slice());
        let p_vanishing = PolynomialOps::sub(&p_lhs_minus_rhs, &p_carry_mul_modulus);

        // Check [lhs(x) - rhs(x) - carry(x) * p(x)] - [witness(x) * (x-2^16)] = 0.
        packed_generic_field_operation::<F, D, FE, PF, D2, P>(
            p_vanishing,
            p_witness_low,
            p_witness_high,
        )
    }

    fn ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
    ) -> Vec<ExtensionTarget<D>> {
        // Get the packed entries.
        let p_a = self.a.register().ext_circuit_vars(vars);
        let p_b = self.b.register().ext_circuit_vars(vars);
        let p_result = self.result.register().ext_circuit_vars(vars);
        let p_carry = self.carry.register().ext_circuit_vars(vars);
        let p_witness_low = self.witness_low.register().ext_circuit_vars(vars);
        let p_witness_high = self.witness_high.register().ext_circuit_vars(vars);

        // Compute the vanishing polynomial:
        //      lhs(x) = sign * (b(x) * result(x) + result(x)) + (1 - sign) * (b(x) * result(x) + a(x))
        //      rhs(x) = sign * a(x) + (1 - sign) * result(x)
        //      lhs(x) - rhs(x) - carry(x) * p(x)
        let p_b_times_res = PolynomialGadget::mul_extension(builder, p_b, p_result);
        let p_equation_lhs = if self.sign {
            PolynomialGadget::add_extension(builder, &p_b_times_res, p_result)
        } else {
            PolynomialGadget::add_extension(builder, &p_b_times_res, p_a)
        };
        let p_equation_rhs = if self.sign { p_a } else { p_result };
        let p_lhs_minus_rhs =
            PolynomialGadget::sub_extension(builder, &p_equation_lhs, p_equation_rhs);
        let p_limbs = PolynomialGadget::constant_extension(
            builder,
            &modulus_field_iter::<F::Extension, P>().collect::<Vec<_>>()[..],
        );
        let mul_times_carry = PolynomialGadget::mul_extension(builder, p_carry, &p_limbs[..]);
        let p_vanishing =
            PolynomialGadget::sub_extension(builder, &p_lhs_minus_rhs, &mul_times_carry);

        // Check [lhs(x) - rhs(x) - carry(x) * p(x)] - [witness(x) * (x-2^16)] = 0.
        ext_circuit_field_operation::<F, D, P>(builder, p_vanishing, p_witness_low, p_witness_high)
    }

    fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) -> Vec<AP::Var> {
        let mut poly_parser = PolynomialParser::new(parser);

        let p_a = self.a.eval(poly_parser.parser);
        let p_b = self.b.eval(poly_parser.parser);
        let p_result = self.result.eval(poly_parser.parser);
        let p_carry = self.carry.eval(poly_parser.parser);

        // Compute the vanishing polynomial:
        //      lhs(x) = sign * (b(x) * result(x) + result(x)) + (1 - sign) * (b(x) * result(x) + a(x))
        //      rhs(x) = sign * a(x) + (1 - sign) * result(x)
        //      lhs(x) - rhs(x) - carry(x) * p(x)
        let p_b_times_res = poly_parser.mul(&p_b, &p_result);
        let p_equation_lhs = if self.sign {
            poly_parser.add(&p_b_times_res, &p_result)
        } else {
            poly_parser.add(&p_b_times_res, &p_a)
        };
        let p_equation_rhs = if self.sign { p_a } else { p_result };

        let p_lhs_minus_rhs = poly_parser.sub(&p_equation_lhs, &p_equation_rhs);
        let p_limbs = poly_parser.constant(&Polynomial::from_iter(modulus_field_iter::<F, P>()));

        let mul_times_carry = poly_parser.mul(&p_carry, &p_limbs);
        let p_vanishing = poly_parser.sub(&p_lhs_minus_rhs, &mul_times_carry);

        let p_witness_low =
            Polynomial::from_coefficients(self.witness_low.eval(poly_parser.parser));
        let p_witness_high =
            Polynomial::from_coefficients(self.witness_high.eval(poly_parser.parser));

        eval_field_operation::<AP, P>(
            &mut poly_parser,
            &p_vanishing,
            &p_witness_low,
            &p_witness_high,
        )
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::rayon;
    use rand::thread_rng;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{ChipStark, StarkParameters};
    use crate::curta::extension::cubic::goldilocks_cubic::GoldilocksCubicParameters;
    use crate::curta::instruction::InstructionSet;
    use crate::curta::parameters::ed25519::Ed25519BaseField;
    use crate::curta::stark::prover::prove;
    use crate::curta::stark::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::curta::stark::verifier::verify_stark_proof;
    use crate::curta::trace::arithmetic::{trace, ArithmeticGenerator};

    #[derive(Clone, Debug, Copy)]
    struct DenTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for DenTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 124;
        const NUM_FREE_COLUMNS: usize = 194;

        type Instruction = InstructionSet<Ed25519BaseField>;
    }

    #[test]
    fn test_ed_den() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = FieldRegister<Ed25519BaseField>;
        type S = ChipStark<DenTest, F, D>;
        type E = GoldilocksCubicParameters;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("stark_proof", log::Level::Debug);

        // build the stark
        let mut builder = StarkBuilder::<DenTest, F, D>::new();

        let a = builder.alloc::<Fp>();
        let b = builder.alloc::<Fp>();
        let sign = false;

        let den_ins = builder.fp_den(&a, &b, sign);
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();

        let chip = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, E, D>(num_rows);

        let p = Ed25519BaseField::modulus();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int: BigUint = rng.gen_biguint(256) % &p;
            let b_int = rng.gen_biguint(256) % &p;
            let handle = handle.clone();
            rayon::spawn(move || {
                let p = Ed25519BaseField::modulus();
                handle.write_field(i as usize, &a_int, a).unwrap();
                handle.write_field(i as usize, &b_int, b).unwrap();
                let res = handle
                    .write_fp_den(i as usize, &a_int, &b_int, sign, den_ins)
                    .unwrap();
                if sign {
                    assert_eq!(res, (a_int * (1u32 + b_int).modpow(&(&p - 2u32), &p)) % &p);
                } else {
                    assert_eq!(
                        res,
                        (a_int * (1u32 + &p - b_int).modpow(&(&p - 2u32), &p)) % &p
                    );
                }
            });
        }
        drop(handle);

        // Generate the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip);
        let proof = prove::<F, C, S, ArithmeticGenerator<F, E, D>, D, 2>(
            stark.clone(),
            &config,
            generator,
            num_rows,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Generate the recursive proof.
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
        verify_stark_proof_circuit::<F, C, S, D, 2>(
            &mut recursive_builder,
            stark,
            virtual_proof,
            &config,
        );
        let recursive_data = recursive_builder.build::<C>();
        let recursive_proof = timed!(
            timing,
            "generate recursive proof",
            plonky2::plonk::prover::prove(
                &recursive_data.prover_only,
                &recursive_data.common,
                rec_pw,
                &mut TimingTree::default(),
            )
            .unwrap()
        );
        timed!(
            timing,
            "verify recursive proof",
            recursive_data.verify(recursive_proof).unwrap()
        );
        timing.print();
    }
}
