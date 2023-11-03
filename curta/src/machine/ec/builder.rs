use core::borrow::Borrow;

use itertools::Itertools;

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

    fn scalar_mul_batch<I, J>(&mut self, points: I, scalars: J) -> Vec<AffinePointRegister<E>>
    where
        I: IntoIterator,
        J: IntoIterator,
        I::Item: Borrow<AffinePointRegister<E>>,
        J::Item: Borrow<ECScalarRegister<E>>,
        Self::Instruction: ECInstructions<E>,
    {
        let nb_bits_log = E::nb_scalar_bits().ilog2();
        assert_eq!(
            E::nb_scalar_bits(),
            1 << nb_bits_log,
            "Scalar size must be a power of 2"
        );

        let cycle_32_size = self.constant(&Self::Field::from_canonical_u32(32));
        let cycle = self.cycle(nb_bits_log as usize);
        let cycle_32 = self.cycle(5);

        let mut results = Vec::new();
        let temp_x_ptr = self.uninit_slice::<FieldRegister<E::BaseField>>();
        let temp_y_ptr = self.uninit_slice::<FieldRegister<E::BaseField>>();
        let x_ptr = self.uninit_slice::<FieldRegister<E::BaseField>>();
        let y_ptr = self.uninit_slice::<FieldRegister<E::BaseField>>();
        let limb_ptr = self.uninit_slice::<ElementRegister>();
        let zero = Time::zero();
        for (i, (point, scalar)) in points.into_iter().zip_eq(scalars).enumerate() {
            let point = point.borrow();
            let scalar = scalar.borrow();

            // Store the EC point.
            let time = Time::constant(256 * i);
            self.store(&temp_x_ptr.get(i), point.x, &time, None);
            self.store(&temp_y_ptr.get(i), point.y, &time, None);

            // Store and the scalar limbs.
            for (j, limb) in scalar.limbs.iter().enumerate() {
                self.store(&limb_ptr.get(i * 8 + j), limb, &zero, Some(cycle_32_size));
            }

            let result = self.alloc_public_ec_point();
            self.free(&x_ptr.get(i), result.x, &zero);
            self.free(&y_ptr.get(i), result.y, &zero);

            results.push(result);
        }
        // Load the elliptic curve point.
        let process_id = self.process_id(cycle.end_bit);

        // Load the scalar limbs.
        let process_id_u32 = self.process_id(cycle_32.end_bit);
        let limb = self.load(&limb_ptr.get_at(process_id_u32), &zero);

        // Decompose the limbs to bits.
        let scalar_bit = self.bit_decomposition(limb, cycle_32.start_bit, cycle_32.end_bit);

        let data = DoubleAddData {
            process_id,
            temp_x_ptr,
            temp_y_ptr,
            result: self.alloc_ec_point(),
            bit: scalar_bit,
            start_bit: cycle.start_bit,
            end_bit: cycle.end_bit,
        };

        // Get `result_next` from the double and add function and store the value at the pointer.
        let result_next = self.double_and_add(&data);
        let end_flag = Some(cycle.end_bit.as_element());
        self.store(&x_ptr.get_at(process_id), result_next.x, &zero, end_flag);
        self.store(&y_ptr.get_at(process_id), result_next.y, &zero, end_flag);

        results
    }

    fn double_and_add(&mut self, data: &DoubleAddData<E>) -> AffinePointRegister<E>
    where
        Self::Instruction: ECInstructions<E>,
    {
        // Keep track of whether res is the identity, which is the point at infinity for some
        // curves.
        //
        // The value starts by being one at the begining of each cycle, and set to zero once the
        // scalar bit is different from zero.
        let is_res_unit = self.alloc::<BitRegister>();
        let scalar_bit = data.bit;
        let end_bit = data.end_bit;
        self.set_to_expression_first_row(&is_res_unit, Self::Field::ONE.into());
        let next_res_unit =
            self.expression(is_res_unit.expr() - scalar_bit.expr() * is_res_unit.expr());
        self.select_next(end_bit, &end_bit, &next_res_unit, &is_res_unit);

        // Load temp.
        let process_id = data.process_id;
        let temp_x_ptr = data.temp_x_ptr.get_at(process_id);
        let temp_y_ptr = data.temp_y_ptr.get_at(process_id);
        let clk = Time::from_element(self.api().clock());
        let temp_x = self.load(&temp_x_ptr, &clk);
        let temp_y = self.load(&temp_y_ptr, &clk);
        let temp = AffinePointRegister::new(temp_x, temp_y);

        // Assign temp_next = temp + temp;
        let not_end_bit = self.expression(data.end_bit.not_expr());
        let temp_next = self.double(&temp);
        self.store(&temp_x_ptr, temp_next.x, &clk.advance(), Some(not_end_bit));
        self.store(&temp_y_ptr, temp_next.y, &clk.advance(), Some(not_end_bit));

        // Calculate res_next = res + temp if scalar_bit is 1, otherwise res_next = res.
        let addend = self.select_ec_point(is_res_unit, &temp_next, &data.result);
        let sum = self.add(&temp, &addend);

        let res_plus_temp = self.select_ec_point(is_res_unit, &temp, &sum);
        let result_next = self.select_ec_point(scalar_bit, &res_plus_temp, &data.result);

        let zero_field = self.zero::<FieldRegister<E::BaseField>>();
        let dummy_point = AffinePointRegister::new(zero_field, zero_field);

        self.select_next_ec_point(end_bit, &dummy_point, &result_next, &data.result);

        result_next
    }
}

impl<E: EllipticCurveAir<B::Parameters>, B: Builder> EllipticCurveBuilder<E> for B {}

#[cfg(test)]
mod tests {
    use log::debug;
    use num::bigint::RandBigInt;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use rand::thread_rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::ArithmeticGenerator;
    use crate::chip::builder::AirBuilder;
    use crate::chip::ec::edwards::ed25519::params::Ed25519;
    use crate::chip::ec::gadget::EllipticCurveWriter;
    use crate::chip::ec::{ECInstruction, EllipticCurve};
    use crate::chip::AirParameters;
    use crate::math::goldilocks::cubic::GoldilocksCubicParameters;
    use crate::maybe_rayon::*;
    use crate::plonky2::stark::config::PoseidonGoldilocksStarkConfig;
    use crate::plonky2::stark::tests::{test_recursive_starky, test_starky};
    use crate::plonky2::stark::Starky;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct Ed25519ScalarMulTest;

    impl AirParameters for Ed25519ScalarMulTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = ECInstruction<Ed25519>;

        const NUM_ARITHMETIC_COLUMNS: usize = 1632;
        const NUM_FREE_COLUMNS: usize = 21;
        const EXTENDED_COLUMNS: usize = 2508;
    }

    #[test]
    fn test_ec_scalar_mul() {
        type F = GoldilocksField;
        type L = Ed25519ScalarMulTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Ed25519;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Ed25519 Scalar mul", log::Level::Debug);

        let mut builder = AirBuilder::<L>::new();
        builder.init_local_memory();

        let points = (0..256)
            .map(|_| builder.alloc_public_ec_point())
            .collect::<Vec<_>>();

        let scalars = (0..256)
            .map(|_| builder.alloc_array_public::<ElementRegister>(8))
            .map(ECScalarRegister::<E>::new)
            .collect::<Vec<_>>();

        let results = builder.scalar_mul_batch(&points, &scalars);

        let (air, trace_data) = builder.build();
        let num_rows = 1 << 16;
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);
        let writer = generator.new_writer();

        let order = E::prime_group_order();

        timed!(
            timing,
            "writing input",
            points
                .par_iter()
                .zip(scalars.par_iter())
                .zip(results.par_iter())
                .for_each(|((point_reg, scalar_reg), result_reg)| {
                    let mut rng = thread_rng();
                    let a = rng.gen_biguint(256);
                    let point = E::ec_generator() * a;
                    let scalar = rng.gen_biguint(256) % &order;
                    let result = &point * &scalar;

                    writer.write_ec_point(point_reg, &point, 0);
                    writer.write_ec_point(result_reg, &result, 0);

                    let mut limb_values = scalar.to_u32_digits();
                    limb_values.resize(8, 0);

                    for (limb_reg, limb) in scalar_reg.limbs.iter().zip_eq(limb_values) {
                        writer.write(&limb_reg, &F::from_canonical_u32(limb), 0);
                    }
                })
        );
        debug!("Wrote input values to public inputs");

        writer.write_global_instructions(&generator.air_data);
        timed!(timing, "generate trace", {
            (0..256usize).for_each(|k| {
                let starting_row = 256 * k;
                for j in 0..256 {
                    let row_index = starting_row + j;
                    writer.write_row_instructions(&generator.air_data, row_index);
                }
            });
        });
        debug!("Generated execution trace");

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        let public_inputs = writer.public.read().unwrap().clone();

        // Generate proof and verify as a stark
        timed!(
            timing,
            "Stark proof and verify",
            test_starky(&stark, &config, &generator, &public_inputs)
        );

        // // Generate recursive proof
        // timed!(
        //     timing,
        //     "Recursive proof generation and verification",
        //     test_recursive_starky(stark, config, generator, &public_inputs)
        // );

        timing.print();
    }
}
