use num::BigUint;

use crate::chip::bool::SelectInstruction;
use crate::chip::builder::AirBuilder;
use crate::chip::ec::edwards::add::EdAddGadget;
use crate::chip::ec::edwards::EdwardsParameters;
use crate::chip::ec::gadget::EllipticCurveWriter;
use crate::chip::ec::point::{AffinePoint, AffinePointRegister};
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::register::FieldRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::trace::writer::TraceWriter;
use crate::chip::utils::biguint_to_bits_le;
use crate::chip::AirParameters;
use crate::math::prelude::*;
use crate::plonky2::field::PrimeField64;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EdDoubleAndAddGadget<E: EdwardsParameters> {
    pub bit: BitRegister,
    pub result: AffinePointRegister<E>,
    pub temp: AffinePointRegister<E>,
    result_next: AffinePointRegister<E>,
    temp_next: AffinePointRegister<E>,
    add_gadget: EdAddGadget<E>,
    double_gadget: EdAddGadget<E>,
    select_x_ins: SelectInstruction<FieldRegister<E::BaseField>>,
    select_y_ins: SelectInstruction<FieldRegister<E::BaseField>>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EdScalarMulGadget<E: EdwardsParameters> {
    pub cyclic_counter: ElementRegister,
    pub double_and_add_gadget: EdDoubleAndAddGadget<E>,
}

impl<L: AirParameters> AirBuilder<L> {
    /// Computes one step of the double-and-add algorithm for scalar multiplication over elliptic
    /// curves. The algorithm the computes the function f(bit, result, temp):
    ///
    /// result = if bit == 1 then result + temp else result
    /// temp = temp + temp
    ///
    /// This function should probably never be used directly and is used in `ed25519_double_and_add`
    pub fn ed_double_and_add<E: EdwardsParameters>(
        &mut self,
        bit: &BitRegister,
        result: &AffinePointRegister<E>,
        temp: &AffinePointRegister<E>,
    ) -> EdDoubleAndAddGadget<E>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        // result = result + temp.
        let add_gadget = self.ed_add(result, temp);

        // temp = temo + temp.
        let double_gadget = self.ed_double(temp);

        // result = if bit == 1 then result + temp else result.
        let select_x_ins = self.select(bit, &add_gadget.result.x, &result.x);
        let select_y_ins = self.select(bit, &add_gadget.result.y, &result.y);
        let result_next = AffinePointRegister::new(select_x_ins.result, select_y_ins.result);

        EdDoubleAndAddGadget {
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

    pub fn ed_scalar_mul<E: EdwardsParameters>(
        &mut self,
        bit: &BitRegister,
        result: &AffinePointRegister<E>,
        temp: &AffinePointRegister<E>,
    ) -> EdScalarMulGadget<E>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        let cyclic_counter = self.alloc::<ElementRegister>();
        let double_and_add_gadget = self.ed_double_and_add(bit, result, temp);

        // Generate a multiplicative subgroup of order 256 (i.e., 2^8).
        let group = L::Field::two_adic_subgroup(8);
        let generator = L::Field::primitive_root_of_unity(8);
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

        EdScalarMulGadget {
            cyclic_counter,
            double_and_add_gadget,
        }
    }
}

impl<F: PrimeField64> TraceWriter<F> {
    pub fn write_ed_double_and_add<E: EdwardsParameters>(
        &self,
        scalar: &BigUint,
        point: &AffinePoint<E>,
        gadget: &EdDoubleAndAddGadget<E>,
        starting_row: usize,
    ) -> AffinePoint<E> {
        let nb_bits = E::nb_scalar_bits();
        let scalar_bits = biguint_to_bits_le(scalar, nb_bits);

        let mut res = E::neutral();
        self.write_ec_point(&gadget.result, &res, starting_row);
        let mut temp = point.clone();
        self.write_ec_point(&gadget.temp, &temp, starting_row);

        for (i, bit) in scalar_bits.iter().enumerate() {
            let f_bit = F::from_canonical_u8(*bit as u8);
            self.write_value(&gadget.bit, &f_bit, starting_row + i);
            let result_plus_temp = &res + &temp;
            self.write_ed_add(&gadget.add_gadget, starting_row + i);
            temp = &temp + &temp;
            self.write_ed_add(&gadget.double_gadget, starting_row + i);

            res = if *bit { result_plus_temp } else { res };

            self.write_instruction(&gadget.select_x_ins, starting_row + i);
            self.write_instruction(&gadget.select_y_ins, starting_row + i);

            if i == nb_bits - 1 {
                break;
            }
            self.write_ec_point(&gadget.result, &res, starting_row + i + 1);
            self.write_ec_point(&gadget.temp, &temp, starting_row + i + 1);
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::edwards::ed25519::{Ed25519, Ed25519BaseField};
    use crate::chip::ec::gadget::EllipticCurveGadget;
    use crate::chip::field::instruction::FpInstruction;

    #[derive(Clone, Debug, Copy)]
    pub struct Ed25519ScalarMulTest;

    impl const AirParameters for Ed25519ScalarMulTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 1504;
        const NUM_FREE_COLUMNS: usize = 2330;
        type Instruction = FpInstruction<Ed25519BaseField>;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_scalar_mul() {
        type F = GoldilocksField;
        type L = Ed25519ScalarMulTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Ed25519;

        let mut builder = AirBuilder::<L>::new();

        let res = builder.alloc_unchecked_ec_point();
        let temp = builder.alloc_unchecked_ec_point();
        let scalar_bit = builder.alloc::<BitRegister>();
        let scalar_mul_gadget = builder.ed_scalar_mul::<E>(&scalar_bit, &res, &temp);

        let (air, _) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(&[]);

        let (tx, rx) = channel();
        let mut rng = thread_rng();

        let mut counter_val = F::ONE;
        let counter_gen = F::primitive_root_of_unity(8);
        let writer = generator.new_writer();
        for j in 0..(1 << 16) {
            writer.write_value(&scalar_mul_gadget.cyclic_counter, &counter_val, j);
            counter_val *= counter_gen;
        }
        for i in 0..256usize {
            let writer = generator.new_writer();
            let handle = tx.clone();
            let gadget = scalar_mul_gadget.clone();
            let a = rng.gen_biguint(256);
            let point = E::generator() * a;
            let scalar = rng.gen_biguint(256);
            rayon::spawn(move || {
                let res = writer.write_ed_double_and_add(
                    &scalar,
                    &point,
                    &gadget.double_and_add_gadget,
                    256 * i,
                );
                assert_eq!(res, point * scalar);
                handle.send(1).unwrap();
            });
        }
        drop(tx);
        for msg in rx.iter() {
            assert!(msg == 1);
        }
        let stark = Starky::<_, { L::num_columns() }>::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
