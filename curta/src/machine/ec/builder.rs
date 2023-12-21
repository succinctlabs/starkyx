use core::borrow::Borrow;

use itertools::Itertools;
use log::debug;
use plonky2::util::log2_ceil;

use super::scalar_mul::DoubleAddData;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::ec::scalar::ECScalarRegister;
use crate::chip::ec::{ECInstructions, EllipticCurveAir};
use crate::chip::field::register::FieldRegister;
use crate::chip::memory::time::Time;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::machine::builder::Builder;
use crate::math::prelude::*;

pub trait EllipticCurveBuilder<E: EllipticCurveAir<Self::Parameters>>: Builder {
    fn alloc_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = self.alloc();
        let y = self.alloc();

        AffinePointRegister::new(x, y)
    }

    fn alloc_public_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = self.alloc_public();
        let y = self.alloc_public();

        AffinePointRegister::new(x, y)
    }

    fn generator(&mut self) -> AffinePointRegister<E> {
        self.api().ec_generator()
    }

    fn select_ec_point(
        &mut self,
        flag: BitRegister,
        true_value: &AffinePointRegister<E>,
        false_value: &AffinePointRegister<E>,
    ) -> AffinePointRegister<E> {
        let result_x = self.select(flag, &true_value.x, &false_value.x);
        let result_y = self.select(flag, &true_value.y, &false_value.y);

        AffinePointRegister::new(result_x, result_y)
    }

    fn select_next_ec_point(
        &mut self,
        flag: BitRegister,
        true_value: &AffinePointRegister<E>,
        false_value: &AffinePointRegister<E>,
        result: &AffinePointRegister<E>,
    ) {
        self.select_next(flag, &true_value.x, &false_value.x, &result.x);
        self.select_next(flag, &true_value.y, &false_value.y, &result.y);
    }

    fn scalar_mul_batch<I, J, K>(&mut self, points: I, scalars: J, results: K)
    where
        I: IntoIterator,
        J: IntoIterator,
        K: IntoIterator,
        I::Item: Borrow<AffinePointRegister<E>>,
        J::Item: Borrow<ECScalarRegister<E>>,
        K::Item: Borrow<AffinePointRegister<E>>,
        Self::Instruction: ECInstructions<E>,
    {
        let nb_scalar_bits = E::nb_scalar_bits();
        let nb_bits_log = nb_scalar_bits.ilog2();
        assert_eq!(
            E::nb_scalar_bits(),
            nb_scalar_bits,
            "Scalar size must be a power of 2"
        );
        assert!(nb_bits_log > 5, "Scalar size must be at least 32 bits");

        let cycle_32_size = self.constant(&Self::Field::from_canonical_u32(32));
        let cycle = self.cycle(nb_bits_log as usize);
        let cycle_32 = self.cycle(5);

        let temp_x_ptr = self.uninit_slice::<FieldRegister<E::BaseField>>();
        let temp_y_ptr = self.uninit_slice::<FieldRegister<E::BaseField>>();
        let x_ptr = self.uninit_slice::<FieldRegister<E::BaseField>>();
        let y_ptr = self.uninit_slice::<FieldRegister<E::BaseField>>();
        let limb_ptr = self.uninit_slice::<ElementRegister>();
        let zero = Time::zero();
        let num_ops = points
            .into_iter()
            .zip_eq(scalars)
            .zip_eq(results)
            .enumerate()
            .map(|(i, ((point, scalar), result))| {
                let point = point.borrow();
                let scalar = scalar.borrow();
                let result = result.borrow();

                // Store the EC point.
                let time = Time::constant(256 * i);
                self.store(&temp_x_ptr.get(i), point.x, &time, None, None, None);
                self.store(&temp_y_ptr.get(i), point.y, &time, None, None, None);

                // Store and the scalar limbs.
                for (j, limb) in scalar.limbs.iter().enumerate() {
                    self.store(
                        &limb_ptr.get(i * 8 + j),
                        limb,
                        &zero,
                        Some(cycle_32_size),
                        None,
                        None,
                    );
                }

                self.free(&x_ptr.get(i), result.x, &zero);
                self.free(&y_ptr.get(i), result.y, &zero);
            })
            .count();

        debug!("AIR degree before padding: {}", num_ops * nb_scalar_bits);
        let degree_log = log2_ceil(num_ops * nb_scalar_bits);
        assert!(degree_log < 31, "AIR degree is too large");
        debug!("AIR degree after padding: {}", 1 << degree_log);
        let num_dummy_ops = (1 << degree_log) / nb_scalar_bits - num_ops;

        // Insert dummy entries where necessary.
        let generator = self.generator();
        let mut one_scalar_limbs = vec![Self::Field::ONE];
        one_scalar_limbs.resize(nb_scalar_bits / 32, Self::Field::ZERO);
        let one_limbs = self.constant_array::<ElementRegister>(&one_scalar_limbs);
        for i in num_ops..(num_ops + num_dummy_ops) {
            let time = Time::constant(256 * i);
            self.store(&temp_x_ptr.get(i), generator.x, &time, None, None, None);
            self.store(&temp_y_ptr.get(i), generator.y, &time, None, None, None);

            // Store and the scalar limbs.
            for (j, limb) in one_limbs.iter().enumerate() {
                self.store(
                    &limb_ptr.get(i * 8 + j),
                    limb,
                    &zero,
                    Some(cycle_32_size),
                    None,
                    None,
                );
            }

            self.free(&x_ptr.get(i), generator.x, &zero);
            self.free(&y_ptr.get(i), generator.y, &zero);
        }

        // Load the elliptic curve point.
        let process_id = self.process_id(nb_scalar_bits, cycle.end_bit);

        // Load the scalar limbs.
        let process_id_u32 = self.process_id(32, cycle_32.end_bit);
        let limb = self.load(&limb_ptr.get_at(process_id_u32), &zero, None, None);

        // Decompose the limbs to bits.
        let scalar_bit = self.bit_decomposition(limb, cycle_32.start_bit, cycle_32.end_bit);

        let data = DoubleAddData {
            process_id,
            temp_x_ptr,
            temp_y_ptr,
            bit: scalar_bit,
            start_bit: cycle.start_bit,
            end_bit: cycle.end_bit,
        };

        // Get `result_next` from the double and add function and store the value at the pointer.
        let result_next = self.double_and_add(&data);
        let end_flag = Some(cycle.end_bit.as_element());
        self.store(
            &x_ptr.get_at(process_id),
            result_next.x,
            &zero,
            end_flag,
            None,
            None,
        );
        self.store(
            &y_ptr.get_at(process_id),
            result_next.y,
            &zero,
            end_flag,
            None,
            None,
        );
    }

    fn double_and_add(&mut self, data: &DoubleAddData<E>) -> AffinePointRegister<E>
    where
        Self::Instruction: ECInstructions<E>,
    {
        // Keep track of whether res is the identity, which is the point at infinity for some
        // curves.
        //
        // The value starts by being '0' at the begining of each cycle, and set to '1' once the
        // scalar bit is different from zero.
        let is_res_valid = self.alloc::<BitRegister>();
        let scalar_bit = data.bit;
        let end_bit = data.end_bit;
        let start_bit = data.start_bit;
        self.set_to_expression_first_row(&is_res_valid, Self::Field::ZERO.into());
        let next_res_valid =
            self.expression(is_res_valid.expr() + scalar_bit.expr() * is_res_valid.not_expr());
        self.select_next(end_bit, &start_bit, &next_res_valid, &is_res_valid);

        // Load temp.
        let process_id = data.process_id;
        let temp_x_ptr = data.temp_x_ptr.get_at(process_id);
        let temp_y_ptr = data.temp_y_ptr.get_at(process_id);
        let clk = Time::from_element(self.clk());
        let temp_x = self.load(&temp_x_ptr, &clk, None, None);
        let temp_y = self.load(&temp_y_ptr, &clk, None, None);
        let temp = AffinePointRegister::new(temp_x, temp_y);

        // Assign temp_next = temp + temp;
        let not_end_bit = self.expression(data.end_bit.not_expr());
        let temp_next = self.double(&temp);
        self.store(
            &temp_x_ptr,
            temp_next.x,
            &clk.advance(),
            Some(not_end_bit),
            None,
            None,
        );
        self.store(
            &temp_y_ptr,
            temp_next.y,
            &clk.advance(),
            Some(not_end_bit),
            None,
            None,
        );

        // Allocate the intermeddiate result.
        let result = self.alloc_ec_point();

        // Calculate res_next = res + temp if scalar_bit is 1, otherwise res_next = res.
        let addend = self.select_ec_point(is_res_valid, &result, &temp_next);
        let sum = self.add(&temp, &addend);

        let res_plus_temp = self.select_ec_point(is_res_valid, &sum, &temp);
        let result_next = self.select_ec_point(scalar_bit, &res_plus_temp, &result);

        let zero_field = self.zero::<FieldRegister<E::BaseField>>();
        let dummy_point = AffinePointRegister::new(zero_field, zero_field);

        // Constrain the intermediate result to be (0, 0) in the first row, and at each transition
        // constrain the result to be equal to `result_next` during each scalar-mul cycle and back
        // to the dummy point (0, 0) at the beginning of each cycle.
        self.set_to_expression_first_row(&result.x, zero_field.expr());
        self.set_to_expression_first_row(&result.y, zero_field.expr());
        self.select_next_ec_point(end_bit, &dummy_point, &result_next, &result);

        result_next
    }
}

impl<E: EllipticCurveAir<B::Parameters>, B: Builder> EllipticCurveBuilder<E> for B {}

#[cfg(test)]
mod tests {
    use log::debug;
    use num::bigint::RandBigInt;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use rand::thread_rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::ec::edwards::ed25519::params::Ed25519;
    use crate::chip::ec::gadget::EllipticCurveAirWriter;
    use crate::chip::ec::{ECInstruction, EllipticCurve};
    use crate::chip::trace::writer::data::AirWriterData;
    use crate::chip::trace::writer::AirWriter;
    use crate::chip::AirParameters;
    use crate::machine::emulated::builder::EmulatedBuilder;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::maybe_rayon::*;
    use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct Ed25519ScalarMulTest;

    impl AirParameters for Ed25519ScalarMulTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ECInstruction<Ed25519>;

        const NUM_ARITHMETIC_COLUMNS: usize = 1632;
        const NUM_FREE_COLUMNS: usize = 19;
        const EXTENDED_COLUMNS: usize = 2502;
    }

    #[test]
    fn test_ec_scalar_mul() {
        type F = GoldilocksField;
        type L = Ed25519ScalarMulTest;
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;
        type E = Ed25519;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Ed25519 Scalar mul", log::Level::Debug);

        let mut builder = EmulatedBuilder::<L>::new();

        let num_ops = 3;

        let points = (0..num_ops)
            .map(|_| builder.alloc_public_ec_point())
            .collect::<Vec<_>>();

        let scalars = (0..num_ops)
            .map(|_| builder.alloc_array_public::<ElementRegister>(8))
            .map(ECScalarRegister::<E>::new)
            .collect::<Vec<_>>();

        let results = (0..num_ops)
            .map(|_| builder.alloc_public_ec_point())
            .collect::<Vec<_>>();

        builder.scalar_mul_batch(&points, &scalars, &results);

        let degree_log = log2_ceil(num_ops * 256);
        let num_rows = 1 << degree_log;
        let stark = builder.build::<C, 2>(1 << degree_log);

        let order = E::prime_group_order();

        // Get thr results
        let ec_data = (0..num_ops)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let a = rng.gen_biguint(256);
                let point = E::ec_generator() * a;
                let scalar = rng.gen_biguint(256) % &order;
                let result = &point * &scalar;
                (point, scalar, result)
            })
            .collect::<Vec<_>>();

        let mut writer_data = AirWriterData::new(&stark.air_data, num_rows);

        let mut writer = writer_data.public_writer();
        timed!(
            timing,
            "writing input",
            points
                .iter()
                .zip(scalars.iter())
                .zip(results.iter())
                .zip(ec_data)
                .for_each(
                    |(((point_reg, scalar_reg), result_reg), (point, scalar, result))| {
                        writer.write_ec_point(point_reg, &point);
                        writer.write_ec_point(result_reg, &result);

                        let mut limb_values = scalar.to_u32_digits();
                        limb_values.resize(8, 0);

                        for (limb_reg, limb) in scalar_reg.limbs.iter().zip_eq(limb_values) {
                            writer.write(&limb_reg, &F::from_canonical_u32(limb));
                        }
                    }
                )
        );

        stark.air_data.write_global_instructions(&mut writer);

        writer_data.chunks_par(256).for_each(|mut chunk| {
            for i in 0..256 {
                let mut writer = chunk.window_writer(i);
                stark.air_data.write_trace_instructions(&mut writer);
            }
        });

        debug!("Generated execution trace");

        let (trace, public) = (writer_data.trace, writer_data.public);

        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();

        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<GoldilocksField, 2>::new(config_rec);

        let (proof_target, public_input) =
            stark.add_virtual_proof_with_pis_target(&mut recursive_builder);
        stark.verify_circuit(&mut recursive_builder, &proof_target, &public_input);

        let data = recursive_builder.build::<Config>();

        let mut pw = PartialWitness::new();

        pw.set_target_arr(&public_input, &public);
        stark.set_proof_target(&mut pw, &proof_target, proof);

        let rec_proof = data.prove(pw).unwrap();
        data.verify(rec_proof).unwrap();

        timing.print();
    }
}
