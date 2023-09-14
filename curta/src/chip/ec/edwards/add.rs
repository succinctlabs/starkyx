use serde::{Serialize, Deserialize};

use super::EdwardsParameters;
use crate::chip::builder::AirBuilder;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::den::FpDenInstruction;
use crate::chip::field::inner_product::FpInnerProductInstruction;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::mul::FpMulInstruction;
use crate::chip::field::mul_const::FpMulConstInstruction;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::AirParameters;
use crate::math::prelude::*;

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdAddGadget<E: EdwardsParameters> {
    p: AffinePointRegister<E>,
    q: AffinePointRegister<E>,
    pub result: AffinePointRegister<E>,
    x3_numerator_ins: FpInnerProductInstruction<E::BaseField>,
    y3_numerator_ins: FpInnerProductInstruction<E::BaseField>,
    x1_mul_y1_ins: FpMulInstruction<E::BaseField>,
    x2_mul_y2_ins: FpMulInstruction<E::BaseField>,
    f_ins: FpMulInstruction<E::BaseField>,
    d_mul_f_ins: FpMulConstInstruction<E::BaseField>,
    x3_ins: FpDenInstruction<E::BaseField>,
    y3_ins: FpDenInstruction<E::BaseField>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn ed_add<E: EdwardsParameters>(
        &mut self,
        p: &AffinePointRegister<E>,
        q: &AffinePointRegister<E>,
    ) -> EdAddGadget<E>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        // Ed25519 Elliptic Curve Addition Formula
        //
        // Given two elliptic curve points (x1, y1) and (x2, y2), compute the sum (x3, y3) with
        //
        // x3 = (x1 * y2 + x2 * y1) / (1 + d * f)
        // y3 = (y1 * y2 + x1 * x2) / (1 - d * f)
        //
        // where f = x1 * x2 * y1 * y2.
        //
        // Reference: https://datatracker.ietf.org/doc/html/draft-josefsson-eddsa-ed25519-02

        let x1 = p.x;
        let x2 = q.x;
        let y1 = p.y;
        let y2 = q.y;

        // x3_numerator = x1 * y2 + x2 * y1.
        let x3_numerator_ins = self.fp_inner_product(&vec![x1, x2], &vec![y2, y1]);

        // y3_numerator = y1 * y2 + x1 * x2.
        let y3_numerator_ins = self.fp_inner_product(&vec![y1, x1], &vec![y2, x2]);

        // f = x1 * x2 * y1 * y2.
        let x1_mul_y1_ins = self.fp_mul(&x1, &y1);
        let x2_mul_y2_ins = self.fp_mul(&x2, &y2);
        let f_ins = self.fp_mul(&x1_mul_y1_ins.result, &x2_mul_y2_ins.result);

        // d * f.
        let d_mul_f_ins = self.fp_mul_const(&f_ins.result, E::D);

        // x3 = x3_numerator / (1 + d * f).
        let x3_ins = self.fp_den(&x3_numerator_ins.result, &d_mul_f_ins.result, true);

        // y3 = y3_numerator / (1 - d * f).
        let y3_ins = self.fp_den(&y3_numerator_ins.result, &d_mul_f_ins.result, false);

        // R = (x3, y3).
        let result = AffinePointRegister::new(x3_ins.result, y3_ins.result);

        EdAddGadget {
            p: *p,
            q: *q,
            result,
            x3_numerator_ins,
            y3_numerator_ins,
            x1_mul_y1_ins,
            x2_mul_y2_ins,
            f_ins,
            d_mul_f_ins,
            x3_ins,
            y3_ins,
        }
    }

    /// Doubles an elliptic curve point `P` on the Ed25519 elliptic curve. Under the hood, the
    /// addition formula is used.
    pub fn ed_double<E: EdwardsParameters>(&mut self, p: &AffinePointRegister<E>) -> EdAddGadget<E>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        self.ed_add(p, p)
    }
}

impl<F: PrimeField64> TraceWriter<F> {
    pub fn write_ed_add<E: EdwardsParameters>(&self, gadget: &EdAddGadget<E>, row_index: usize) {
        self.write_instruction(&gadget.x3_numerator_ins, row_index);
        self.write_instruction(&gadget.y3_numerator_ins, row_index);
        self.write_instruction(&gadget.x1_mul_y1_ins, row_index);
        self.write_instruction(&gadget.x2_mul_y2_ins, row_index);
        self.write_instruction(&gadget.f_ins, row_index);
        self.write_instruction(&gadget.d_mul_f_ins, row_index);
        self.write_instruction(&gadget.x3_ins, row_index);
        self.write_instruction(&gadget.y3_ins, row_index);
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::edwards::ed25519::{Ed25519, Ed25519BaseField};
    use crate::chip::ec::gadget::{EllipticCurveGadget, EllipticCurveWriter};
    use crate::chip::field::instruction::FpInstruction;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    pub struct Ed25519AddTest;

    impl AirParameters for Ed25519AddTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 800;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 1209;
        type Instruction = FpInstruction<Ed25519BaseField>;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_ed25519_add() {
        type L = Ed25519AddTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Ed25519;

        let mut builder = AirBuilder::<L>::new();

        let p = builder.alloc_ec_point();
        let q = builder.alloc_ec_point();

        let _gadget = builder.ed_add::<E>(&p, &q);

        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data);

        let base = E::generator();
        let mut rng = thread_rng();
        let a = rng.gen_biguint(256);
        let b = rng.gen_biguint(256);
        let p_int = &base * &a;
        let q_int = &base * &b;
        let writer = generator.new_writer();
        (0..L::num_rows()).into_par_iter().for_each(|i| {
            writer.write_ec_point(&p, &p_int, i);
            writer.write_ec_point(&q, &q_int, i);
            writer.write_row_instructions(&generator.air_data, i);
        });

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(L::num_rows());

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &[]);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &[]);
    }
}
