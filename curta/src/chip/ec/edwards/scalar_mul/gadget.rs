use serde::{Serialize, Deserialize};

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::builder::AirBuilder;
use crate::chip::ec::edwards::add::EdAddGadget;
use crate::chip::ec::edwards::EdwardsParameters;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::instruction::cycle::Cycle;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::AirParameters;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct EdDoubleAndAddGadget<E: EdwardsParameters> {
    pub bit: BitRegister,
    pub result: AffinePointRegister<E>,
    pub temp: AffinePointRegister<E>,
    result_next: AffinePointRegister<E>,
    temp_next: AffinePointRegister<E>,
    add_gadget: EdAddGadget<E>,
    double_gadget: EdAddGadget<E>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct EdScalarMulGadget<F, E: EdwardsParameters> {
    pub cycle: Cycle<F>,
    pub double_and_add_gadget: EdDoubleAndAddGadget<E>,
}

#[allow(clippy::misnamed_getters)]
impl<F, E: EdwardsParameters> EdScalarMulGadget<F, E> {
    pub fn result(&self) -> AffinePointRegister<E> {
        self.double_and_add_gadget.result_next
    }

    pub fn temp(&self) -> AffinePointRegister<E> {
        self.double_and_add_gadget.temp
    }
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
        let select_x = self.select(bit, &add_gadget.result.x, &result.x);
        let select_y = self.select(bit, &add_gadget.result.y, &result.y);
        let result_next = AffinePointRegister::new(select_x, select_y);

        EdDoubleAndAddGadget {
            bit: *bit,
            result: *result,
            temp: *temp,
            result_next,
            temp_next: double_gadget.result,
            add_gadget,
            double_gadget,
            // select_x_ins,
            // select_y_ins,
        }
    }

    pub fn ed_scalar_mul<E: EdwardsParameters>(
        &mut self,
        bit: &BitRegister,
        result: &AffinePointRegister<E>,
        temp: &AffinePointRegister<E>,
    ) -> EdScalarMulGadget<L::Field, E>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        // Create a cycle of size 256
        let cycle = self.cycle(8);
        let double_and_add_gadget = self.ed_double_and_add(bit, result, temp);

        // Copy over the result of the double and add step to the next row for every row but not for
        // every 256th row. By doing this trick, we can compute multiple scalar multiplications
        // in a single STARK.
        let result = double_and_add_gadget.result;
        let result_next = double_and_add_gadget.result_next;
        let temp = double_and_add_gadget.temp;
        let temp_next = double_and_add_gadget.temp_next;

        // Note that result and result_next live on the same row.
        // if log_generator(cursor[LOCAL]) % 2^8 == 0 then result[NEXT] <= result_next[LOCAL].

        let flag_bit = cycle.start_bit.next().expr::<L::Field>();

        let result_x_next_val = flag_bit.clone() * result.x.next().expr()
            + (ArithmeticExpression::one() - flag_bit.clone()) * result_next.x.expr();
        self.set_to_expression_transition(&result.x.next(), result_x_next_val);

        let result_y_next_val = flag_bit.clone() * result.y.next().expr()
            + (ArithmeticExpression::one() - flag_bit.clone()) * result_next.y.expr();
        self.set_to_expression_transition(&result.y.next(), result_y_next_val);

        let temp_x_next_val = flag_bit.clone() * temp.x.next().expr()
            + (ArithmeticExpression::one() - flag_bit.clone()) * temp_next.x.expr();
        self.set_to_expression_transition(&temp.x.next(), temp_x_next_val);

        let temp_y_next_val = flag_bit.clone() * temp.y.next().expr()
            + (ArithmeticExpression::one() - flag_bit.clone()) * temp_next.y.expr();
        self.set_to_expression_transition(&temp.y.next(), temp_y_next_val);

        EdScalarMulGadget {
            cycle,
            double_and_add_gadget,
        }
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use rand::thread_rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::edwards::ed25519::{Ed25519, Ed25519BaseField};
    use crate::chip::ec::gadget::{EllipticCurveGadget, EllipticCurveWriter};
    use crate::chip::field::instruction::FpInstruction;
    use crate::chip::utils::biguint_to_bits_le;
    use crate::math::prelude::*;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    pub struct Ed25519ScalarMulTest;

    impl AirParameters for Ed25519ScalarMulTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 1504;
        const NUM_FREE_COLUMNS: usize = 72;
        const EXTENDED_COLUMNS: usize = 2265;
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

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("Ed25519 Scalar mul", log::Level::Debug);

        let mut builder = AirBuilder::<L>::new();

        let res = builder.alloc_unchecked_ec_point();
        let temp = builder.alloc_unchecked_ec_point();
        let scalar_bit = builder.alloc::<BitRegister>();
        let _scalar_mul_gadget = builder.ed_scalar_mul::<E>(&scalar_bit, &res, &temp);

        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data);

        let writer = generator.new_writer();
        let nb_bits = E::nb_scalar_bits();
        timed!(timing, "generate trace", {
            (0..256usize).into_par_iter().for_each(|k| {
                let starting_row = 256 * k;
                // let writer = generator.new_writer();
                let mut rng = thread_rng();
                // let handle = tx.clone();
                let a = rng.gen_biguint(256);
                let point = E::generator() * a;
                writer.write_ec_point(&res, &E::neutral(), starting_row);
                writer.write_ec_point(&temp, &point, starting_row);
                let scalar = rng.gen_biguint(256);
                let scalar_bits = biguint_to_bits_le(&scalar, nb_bits);
                for (i, bit) in scalar_bits.iter().enumerate() {
                    let f_bit = F::from_canonical_u8(*bit as u8);
                    writer.write(&scalar_bit, &f_bit, starting_row + i);
                    writer.write_row_instructions(&generator.air_data, starting_row + i);
                }
            });
        });
        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        timed!(
            timing,
            "Stark proof and verify",
            test_starky(&stark, &config, &generator, &[])
        );

        // Generate recursive proof
        timed!(
            timing,
            "Recursive proof generation and verification",
            test_recursive_starky(stark, config, generator, &[])
        );

        timing.print();
    }
}
