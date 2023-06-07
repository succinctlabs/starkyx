use super::add::Ed25519AddGadget;
use super::*;
use crate::curta::bool::SelectInstruction;
use crate::curta::builder::StarkBuilder;
use crate::curta::chip::StarkParameters;
use crate::curta::instruction::FromInstructionSet;
use crate::curta::parameters::EdwardsParameters;
use crate::curta::polynomial::Polynomial;
use crate::curta::register::{BitRegister, ElementRegister};
use crate::curta::utils::biguint_to_bits_le;

#[derive(Clone)]
#[allow(dead_code)]
pub struct Ed25519DoubleAndAddGadget<E: EdwardsParameters> {
    bit: BitRegister,
    result: AffinePointRegister<E>,
    temp: AffinePointRegister<E>,
    result_next: AffinePointRegister<E>,
    temp_next: AffinePointRegister<E>,
    add_gadget: Ed25519AddGadget<E>,
    double_gadget: Ed25519AddGadget<E>,
    select_x_ins: SelectInstruction<FieldRegister<E::BaseField>>,
    select_y_ins: SelectInstruction<FieldRegister<E::BaseField>>,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct Ed25519ScalarMulGadget<E: EdwardsParameters> {
    cyclic_counter: ElementRegister,
    double_and_add_gadget: Ed25519DoubleAndAddGadget<E>,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    /// Computes one step of the double-and-add algorithm for scalar multiplication over elliptic
    /// curves. The algorithm the computes the function f(bit, result, temp):
    ///
    /// result = if bit == 1 then result + temp else result
    /// temp = temp + temp
    ///
    /// This function should probably never be used directly and is used in `ed25519_double_and_add`
    fn ed25519_double_and_add<E: EdwardsParameters>(
        &mut self,
        bit: &BitRegister,
        result: &AffinePointRegister<E>,
        temp: &AffinePointRegister<E>,
    ) -> Ed25519DoubleAndAddGadget<E>
    where
        L::Instruction: FromInstructionSet<E::BaseField>,
    {
        // result = result + temp.
        let add_gadget = self.ed25519_add(result, temp);

        // temp = temo + temp.
        let double_gadget = self.ed25519_double(temp);

        // result = if bit == 1 then result + temp else result.
        let select_x_ins = self.select(bit, &add_gadget.result.x, &result.x);
        let select_y_ins = self.select(bit, &add_gadget.result.y, &result.y);
        let result_next = AffinePointRegister::new(select_x_ins.result, select_y_ins.result);

        Ed25519DoubleAndAddGadget {
            bit: *bit,
            result: *result,
            temp: *temp,
            result_next,
            temp_next: double_gadget.result,
            add_gadget,
            double_gadget,
            select_x_ins,
            select_y_ins,
        }
    }

    pub fn ed25519_scalar_mul<E: EdwardsParameters>(
        &mut self,
        bit: &BitRegister,
        result: &AffinePointRegister<E>,
        temp: &AffinePointRegister<E>,
    ) -> Ed25519ScalarMulGadget<E>
    where
        L::Instruction: FromInstructionSet<E::BaseField>,
    {
        let cyclic_counter = self.alloc::<ElementRegister>();
        let double_and_add_gadget = self.ed25519_double_and_add(bit, result, temp);

        // Generate a multiplicative subgroup of order 256 (i.e., 2^8).
        let group = F::two_adic_subgroup(8);
        let generator = F::primitive_root_of_unity(8);
        let generator_inv = group[group.len() - 1];
        debug_assert_eq!(generator, group[1]);

        // Copy over the result of the double and add step to the next row for every row but not for
        // every 256th row. By doing this trick, we can compute multiple scalar multiplications
        // in a single STARK.
        let result = double_and_add_gadget.result;
        let result_next = double_and_add_gadget.result_next;
        let temp = double_and_add_gadget.temp;
        let temp_next = double_and_add_gadget.temp_next;

        // Note that result and result_next live on the same row.
        // if log_generator(cursor[LOCAL]) % 2^8 == 0 then result[NEXT] <= result_next[LOCAL].
        let result_x_copy_constraint = (cyclic_counter.expr() - generator_inv)
            * (result.x.next().expr() - result_next.x.expr());
        self.assert_expression_zero(result_x_copy_constraint);
        let result_y_copy_constraint = (cyclic_counter.expr() - generator_inv)
            * (result.y.next().expr() - result_next.y.expr());
        self.assert_expression_zero(result_y_copy_constraint);

        // Note that temp and temp_next live on the same row.
        // if log_generator(cursor[LOCAL]) % 2^8 == 0 then temp[NEXT] <= temp_next[LOCAL]
        let temp_x_copy_constraint =
            (cyclic_counter.expr() - generator_inv) * (temp.x.next().expr() - temp_next.x.expr());
        self.assert_expression_zero(temp_x_copy_constraint);
        let temp_y_copy_constraint =
            (cyclic_counter.expr() - generator_inv) * (temp.y.next().expr() - temp_next.y.expr());
        self.assert_expression_zero(temp_y_copy_constraint);

        // cursor[NEXT] = cursor[LOCAL] * generator
        self.assert_expressions_equal(
            cyclic_counter.next().expr(),
            cyclic_counter.expr() * generator,
        );

        Ed25519ScalarMulGadget {
            cyclic_counter,
            double_and_add_gadget,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    /// Writes the double-and-add algorithm for scalar multiplication to the trace. Assumes the
    /// scalar is already reduced modulo the order of the curve group.
    #[allow(non_snake_case)]
    pub fn write_ed25519_double_and_add<E: EdwardsParameters>(
        &self,
        starting_row: usize,
        scalar: &BigUint,
        point: &AffinePoint<E>,
        double_and_add_gadget: Ed25519DoubleAndAddGadget<E>,
    ) -> Result<AffinePoint<E>> {
        let nb_bits = E::nb_scalar_bits();
        let scalar_bits = biguint_to_bits_le(scalar, nb_bits);

        let mut res = E::neutral();
        self.write_ec_point(starting_row, &res, &double_and_add_gadget.result)?;
        let mut temp = point.clone();
        self.write_ec_point(starting_row, &temp, &double_and_add_gadget.temp)?;

        for (i, bit) in scalar_bits.iter().enumerate() {
            self.write_bit(starting_row + i, *bit, &double_and_add_gadget.bit)?;
            let result_plus_temp = self.write_ed25519_add(
                starting_row + i,
                &res,
                &temp,
                double_and_add_gadget.add_gadget.clone(),
            )?;
            temp = self.write_ed25519_double(
                starting_row + i,
                &temp,
                double_and_add_gadget.double_gadget.clone(),
            )?;
            res = if *bit { result_plus_temp } else { res };
            let res_x_field_vec =
                Polynomial::from_biguint_field(&res.x, 16, E::BaseField::NB_LIMBS).into_vec();
            let res_y_field_vec =
                Polynomial::from_biguint_field(&res.y, 16, E::BaseField::NB_LIMBS).into_vec();
            self.write(
                starting_row + i,
                double_and_add_gadget.select_x_ins,
                res_x_field_vec,
            )?;
            self.write(
                starting_row + i,
                double_and_add_gadget.select_y_ins,
                res_y_field_vec,
            )?;
            if i == nb_bits - 1 {
                break;
            }
            self.write_ec_point(starting_row + i + 1, &res, &double_and_add_gadget.result)?;
            self.write_ec_point(starting_row + i + 1, &temp, &double_and_add_gadget.temp)?;
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{ChipStark, StarkParameters};
    use crate::curta::extension::cubic::goldilocks_cubic::GoldilocksCubicParameters;
    use crate::curta::instruction::InstructionSet;
    use crate::curta::parameters::ed25519::{Ed25519, Ed25519BaseField};
    use crate::curta::stark::prover::prove;
    use crate::curta::stark::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::curta::stark::verifier::verify_stark_proof;
    use crate::curta::trace::arithmetic::{trace, ArithmeticGenerator};

    #[derive(Clone, Debug, Copy)]
    pub struct Ed25519ScalarMulTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for Ed25519ScalarMulTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 1504;
        const NUM_FREE_COLUMNS: usize = 2330; //2 + 2 * 2 * 16;
        type Instruction = InstructionSet<Ed25519BaseField>;
    }

    #[test]
    fn test_scalar_mul() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519;
        type L = Ed25519ScalarMulTest;
        type S = ChipStark<L, F, D>;
        type CUB = GoldilocksCubicParameters;
        let _ = env_logger::builder().is_test(true).try_init();

        // Build the stark.
        let mut builder = StarkBuilder::<L, F, D>::new();
        let res = builder.alloc_unchecked_ec_point::<E>();
        let temp = builder.alloc_unchecked_ec_point::<E>();
        let scalar_bit = builder.alloc::<BitRegister>();
        let scalar_mul_gadget = builder.ed25519_scalar_mul(&scalar_bit, &res, &temp);
        builder.write_data(&scalar_bit).unwrap();
        builder
            .write_data(&scalar_mul_gadget.cyclic_counter)
            .unwrap();
        builder.write_ec_point(&res).unwrap();
        builder.write_ec_point(&temp).unwrap();
        let chip = builder.build();

        // Generate the trace.
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, CUB, D>(num_rows);
        let mut timing = TimingTree::new("Ed_Add row", log::Level::Debug);
        let mut counter_val = F::ONE;
        let counter_gen = F::primitive_root_of_unity(8);
        for j in 0..(1 << 16) {
            handle
                .write_data(j, scalar_mul_gadget.cyclic_counter, vec![counter_val])
                .unwrap();
            counter_val *= counter_gen;
        }
        for i in 0..256usize {
            let handle = handle.clone();
            let scalar_mul_gadget = scalar_mul_gadget.clone();
            rayon::spawn(move || {
                let mut rng = thread_rng();
                let a = rng.gen_biguint(256);
                let point = E::generator() * a;
                let scalar = rng.gen_biguint(256);
                let res = handle
                    .write_ed25519_double_and_add(
                        256 * i,
                        &scalar,
                        &point,
                        scalar_mul_gadget.clone().double_and_add_gadget,
                    )
                    .unwrap();
                assert_eq!(res, point * scalar);
            });
        }
        drop(handle);

        // Generate the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip);
        let proof = prove::<F, C, S, ArithmeticGenerator<F, CUB, D>, D, 2>(
            stark.clone(),
            &config,
            generator,
            num_rows,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        // Verify proof as a stark
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

    // A function for testing wrong proof that doesn't connect the two rows of the
    // trace correctly (this checks the counter constraints)
    #[allow(non_snake_case)]
    pub fn write_ed_double_add_test_transition<
        F: RichField + Extendable<D>,
        const D: usize,
        E: EdwardsParameters,
    >(
        handle: &TraceWriter<F, D>,
        starting_row: usize,
        scalar: &BigUint,
        point: &AffinePoint<E>,
        chip_data: Ed25519DoubleAndAddGadget<E>,
    ) -> Result<AffinePoint<E>> {
        let num_bits = E::nb_scalar_bits();
        let scalar_bits = biguint_to_bits_le(scalar, num_bits);

        let mut res = E::neutral();
        handle.write_ec_point(starting_row, &res, &chip_data.result)?;
        let mut temp = point.clone();
        handle.write_ec_point(starting_row, &temp, &chip_data.temp)?;

        for (i, bit) in scalar_bits.iter().enumerate() {
            handle.write_bit(starting_row + i, *bit, &chip_data.bit)?;
            let result_plus_temp = handle.write_ed25519_add(
                starting_row + i,
                &res,
                &temp,
                chip_data.add_gadget.clone(),
            )?;
            temp = handle.write_ed25519_double(
                starting_row + i,
                &temp,
                chip_data.double_gadget.clone(),
            )?;
            res = if *bit { result_plus_temp } else { res };
            let res_x_field_vec =
                Polynomial::from_biguint_field(&res.x, 16, E::BaseField::NB_LIMBS).into_vec();
            let res_y_field_vec =
                Polynomial::from_biguint_field(&res.y, 16, E::BaseField::NB_LIMBS).into_vec();
            handle.write(starting_row + i, chip_data.select_x_ins, res_x_field_vec)?;
            handle.write(starting_row + i, chip_data.select_y_ins, res_y_field_vec)?;

            if i == num_bits - 1 {
                break;
            }
            temp = &temp + &temp;
            res = &res + &res;
            handle.write_ec_point(starting_row + i + 1, &res, &chip_data.result)?;
            handle.write_ec_point(starting_row + i + 1, &temp, &chip_data.temp)?;
        }
        Ok(res)
    }

    #[test]
    fn test_failed_transition_scalar_mul_proof() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519;
        type L = Ed25519ScalarMulTest;
        type S = ChipStark<L, F, D>;
        type CUB = GoldilocksCubicParameters;
        let _ = env_logger::builder().is_test(true).try_init();

        // Build the stark.
        let mut builder = StarkBuilder::<L, F, D>::new();
        let res = builder.alloc_unchecked_ec_point::<E>();
        let temp = builder.alloc_unchecked_ec_point::<E>();
        let bit = builder.alloc::<BitRegister>();
        builder.write_data(&bit).unwrap();
        let scalar_mul_gadget = builder.ed25519_scalar_mul(&bit, &res, &temp);
        builder
            .write_data(&scalar_mul_gadget.cyclic_counter)
            .unwrap();
        builder.write_ec_point(&res).unwrap();
        builder.write_ec_point(&temp).unwrap();
        let chip = builder.build();

        // Generate the trace.
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, CUB, D>(num_rows);
        let mut counter_val = F::ONE;
        let counter_gen = F::primitive_root_of_unity(8);
        for j in 0..(1 << 16) {
            handle
                .write_data(j, scalar_mul_gadget.cyclic_counter, vec![counter_val])
                .unwrap();
            counter_val *= counter_gen;
        }
        for i in 0..256usize {
            let handle = handle.clone();
            let scalar_mul_gadget = scalar_mul_gadget.clone();
            rayon::spawn(move || {
                let mut rng = thread_rng();
                let a = rng.gen_biguint(256);
                let point = E::generator() * a;
                let scalar = rng.gen_biguint(256);
                let res = write_ed_double_add_test_transition(
                    &handle,
                    256 * i,
                    &scalar,
                    &point,
                    scalar_mul_gadget.clone().double_and_add_gadget,
                )
                .unwrap();
                assert_ne!(res, point * scalar);
            });
        }
        drop(handle);

        // Verify proof as a stark.
        // Generate the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip);
        let proof = prove::<F, C, S, ArithmeticGenerator<F, CUB, D>, D, 2>(
            stark.clone(),
            &config,
            generator,
            num_rows,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        let res = verify_stark_proof(stark.clone(), proof, &config);
        assert!(res.is_err());
    }
}
