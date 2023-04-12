use super::add::{EcAddData, FromEdwardsAdd};
use super::*;
use crate::arithmetic::bool::Selector;
use crate::arithmetic::builder::ChipBuilder;
use crate::arithmetic::chip::ChipParameters;
use crate::arithmetic::polynomial::Polynomial;
use crate::arithmetic::register::BitRegister;
use crate::arithmetic::util::biguint_to_bits_le;

#[derive(Clone, Copy)]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct EdScalarMulData<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize> {
    result: AffinePointRegister<E, N_LIMBS>,
    result_next: AffinePointRegister<E, N_LIMBS>,
    temp: AffinePointRegister<E, N_LIMBS>,
    temp_next: AffinePointRegister<E, N_LIMBS>,
    scalar_bit: BitRegister,
    add: EcAddData<E, N_LIMBS>,
    double: EcAddData<E, N_LIMBS>,
    selector_x: Selector<FieldRegister<E::FieldParam, N_LIMBS>>,
    selector_y: Selector<FieldRegister<E::FieldParam, N_LIMBS>>,
}

impl<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize> EdScalarMulData<E, N_LIMBS> {
    pub const fn num_ed_scalar_mul_witness_columns() -> usize {
        2 * N_LIMBS + 2 * EcAddData::<E, N_LIMBS>::num_ed_add_witness_columns()
    }

    pub const fn num_ed_scalar_mul_columns() -> usize {
        1 + 2 * 2 * N_LIMBS + Self::num_ed_scalar_mul_witness_columns()
    }
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
    /// This function implements the double-and-add algorithm for scalar multiplication.
    ///
    /// The function performs the following operation:
    ///     Add a write option for the bits to scalar_bit
    ///     Asserts Result_next = if scalar_bit == 1 then result + temp else Result
    ///     Asserts temp_next = temp + temp
    ///
    pub fn ed_double_and_add<E: EdwardsParameters<N>, const N: usize>(
        &mut self,
        scalar_bit: &BitRegister,
        result: &AffinePointRegister<E, N>,
        result_next: &AffinePointRegister<E, N>,
        temp: &AffinePointRegister<E, N>,
        temp_next: &AffinePointRegister<E, N>,
    ) -> Result<EdScalarMulData<E, N>>
    where
        L::Instruction: FromEdwardsAdd<E, N> + From<Selector<FieldRegister<E::FieldParam, N>>>,
    {
        let result_plus_temp = self.alloc_ec_point()?;
        self.write_data(scalar_bit)?;

        let add = self.ed_add(&result, &temp, &result_plus_temp)?;
        let double = self.ed_double(&temp, &temp_next)?;

        let selector_x =
            self.selector(scalar_bit, &result_plus_temp.x, &result.x, &result_next.x)?;
        let selector_y =
            self.selector(scalar_bit, &result_plus_temp.y, &result.y, &result_next.y)?;

        Ok(EdScalarMulData {
            result: *result,
            result_next: *result_next,
            temp: *temp,
            temp_next: *temp_next,
            scalar_bit: scalar_bit.clone(),
            add,
            double,
            selector_x,
            selector_y,
        })
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceHandle<F, D> {
    /// Writes the double-and-add algorithm for scalar multiplication to the trace.
    ///
    /// Assumes the scalar is already reduced modulo the order of the curve group
    #[allow(non_snake_case)]
    pub fn write_ed_double_and_add<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize>(
        &self,
        starting_row: usize,
        scalar: &BigUint,
        result: &AffinePoint<E, N_LIMBS>,
        temp: &AffinePoint<E, N_LIMBS>,
        chip_data: EdScalarMulData<E, N_LIMBS>,
    ) -> Result<(AffinePoint<E, N_LIMBS>, AffinePoint<E, N_LIMBS>)> {
        let num_bits = E::num_scalar_bits();
        let scalar_bits = biguint_to_bits_le(scalar, num_bits);

        let mut res = result.clone();
        let mut temp = temp.clone();

        for (i, bit) in scalar_bits.iter().enumerate() {
            self.write_bit(starting_row + i, *bit, &chip_data.scalar_bit)?;
            let result_plus_temp =
                self.write_ed_add(starting_row + i, &res, &temp, chip_data.add)?;
            temp = self.write_ed_double(starting_row + i, &temp, chip_data.double)?;
            res = if *bit { result_plus_temp } else { res };
            let res_x_field_vec = Polynomial::from_biguint_field(&res.x, 16, N_LIMBS).into_vec();
            let res_y_field_vec = Polynomial::from_biguint_field(&res.y, 16, N_LIMBS).into_vec();
            self.write(starting_row + i, chip_data.selector_x, res_x_field_vec)?;
            self.write(starting_row + i, chip_data.selector_y, res_y_field_vec)?;
            // for testing
            if i == num_bits - 1 {
                break;
            }
            self.write_ec_point(starting_row + i + 1, &res, &chip_data.result)?;
            self.write_ec_point(starting_row + i + 1, &temp, &chip_data.temp)?;
        }
        Ok((res, temp))
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::arithmetic::builder::ChipBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
    use crate::arithmetic::ec::edwards::instructions::EdWardsMicroInstruction;
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    pub struct EdScalarMulTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for EdScalarMulTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 2 * 2 * 16
            + EdScalarMulData::<Ed25519Parameters, 16>::num_ed_scalar_mul_columns()
            - 1
            - 3 * 2 * 2 * 16;
        const NUM_FREE_COLUMNS: usize = 1 + 2 * 2 * 16;

        type Instruction = EdWardsMicroInstruction<Ed25519Parameters, 16>;
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_ed_double_and_add() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519Parameters;
        type S = TestStark<EdScalarMulTest, F, D>;
        type FieldPar = <E as EllipticCurveParameters<16>>::FieldParam;

        let _ = env_logger::builder().is_test(true).try_init();
        // build the stark
        let mut builder = ChipBuilder::<EdScalarMulTest, F, D>::new();

        let n_limbs = 16;

        //let res = builder.alloc_ec_point::<E, 16>().unwrap();
        //let temp = builder.alloc_ec_point::<E, 16>().unwrap();
        let res_x = FieldRegister::<FieldPar, 16>::from_raw_register(
            builder.get_local_memory(n_limbs).unwrap(),
        );
        let res_y = FieldRegister::<FieldPar, 16>::from_raw_register(
            builder.get_local_memory(n_limbs).unwrap(),
        );
        let res = AffinePointRegister::<E, 16>::from_field_registers(res_x, res_y);
        let temp_x = FieldRegister::<FieldPar, 16>::from_raw_register(
            builder.get_local_memory(n_limbs).unwrap(),
        );
        let temp_y = FieldRegister::<FieldPar, 16>::from_raw_register(
            builder.get_local_memory(n_limbs).unwrap(),
        );
        let temp = AffinePointRegister::<E, 16>::from_field_registers(temp_x, temp_y);

        let res_next = builder.alloc_ec_point::<E, 16>().unwrap();
        let temp_next = builder.alloc_ec_point::<E, 16>().unwrap();

        let bit = builder.alloc_local::<BitRegister>().unwrap();

        let ed_data = builder
            .ed_double_and_add(&bit, &res, &res_next, &temp, &temp_next)
            .unwrap();
        builder.write_ec_point(&res).unwrap();
        builder.write_ec_point(&temp).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let mut timing = TimingTree::new("Ed_Add row", log::Level::Debug);

        let trace = timed!(timing, "generate trace", {
            for i in 0..256usize {
                let handle = handle.clone();
                rayon::spawn(move || {
                    let mut rng = thread_rng();
                    let a = rng.gen_biguint(256);
                    let scalar = rng.gen_biguint(256);
                    let temp_int = E::generator() * a;
                    let res_int = AffinePoint::neutral();
                    handle
                        .write_ec_point(256 * i, &temp_int, &ed_data.temp)
                        .unwrap();
                    handle
                        .write_ec_point(256 * i, &res_int, &ed_data.result)
                        .unwrap();

                    let (_res, _temp) = handle
                        .write_ed_double_and_add(
                            256 * i,
                            &scalar,
                            &res_int,
                            &temp_int,
                            ed_data,
                        )
                        .unwrap();
                });
            }
            drop(handle);

            generator.generate_trace(&chip, num_rows as usize).unwrap()
        });

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);

        // Verify proof as a stark
        let proof = timed!(
            timing,
            "generate stark proof",
            prove::<F, C, S, D>(
                stark.clone(),
                &config,
                trace,
                [],
                &mut TimingTree::default(),
            )
            .unwrap()
        );
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
        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }
}
