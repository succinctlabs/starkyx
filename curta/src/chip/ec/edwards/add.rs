use super::{EdwardsCurve, EdwardsParameters};
use crate::chip::builder::AirBuilder;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    pub fn ed_add<E: EdwardsParameters>(
        &mut self,
        p: &AffinePointRegister<EdwardsCurve<E>>,
        q: &AffinePointRegister<EdwardsCurve<E>>,
    ) -> AffinePointRegister<EdwardsCurve<E>>
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
        let x3_numerator = self.fp_inner_product(&[x1, x2], &[y2, y1]);

        // y3_numerator = y1 * y2 + x1 * x2.
        let y3_numerator = self.fp_inner_product(&[y1, x1], &[y2, x2]);

        // f = x1 * x2 * y1 * y2.
        let x1_mul_y1 = self.fp_mul(&x1, &y1);
        let x2_mul_y2 = self.fp_mul(&x2, &y2);
        let f = self.fp_mul(&x1_mul_y1, &x2_mul_y2);

        // d * f.
        let d_mul_f = self.fp_mul_const(&f, E::D);

        // x3 = x3_numerator / (1 + d * f).
        let x3_ins = self.fp_den(&x3_numerator, &d_mul_f, true);

        // y3 = y3_numerator / (1 - d * f).
        let y3_ins = self.fp_den(&y3_numerator, &d_mul_f, false);

        // R = (x3, y3).
        AffinePointRegister::new(x3_ins.result, y3_ins.result)
    }

    /// Doubles an elliptic curve point `P` on the Ed25519 elliptic curve. Under the hood, the
    /// addition formula is used.
    pub fn ed_double<E: EdwardsParameters>(
        &mut self,
        p: &AffinePointRegister<EdwardsCurve<E>>,
    ) -> AffinePointRegister<EdwardsCurve<E>>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        self.ed_add(p, p)
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::edwards::ed25519::params::{Ed25519, Ed25519BaseField};
    use crate::chip::ec::gadget::{EllipticCurveGadget, EllipticCurveWriter};
    use crate::chip::ec::EllipticCurve;
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
    }

    #[test]
    fn test_ed25519_add() {
        type L = Ed25519AddTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Ed25519;

        let mut builder = AirBuilder::<L>::new();

        let p_pub = builder.alloc_public_ec_point();
        let q_pub = builder.alloc_public_ec_point();
        let _gadget_pub = builder.ec_add::<E>(&p_pub, &q_pub);

        let p = builder.alloc_ec_point();
        let q = builder.alloc_ec_point();
        let _gadget = builder.ec_add::<E>(&p, &q);

        let num_rows = 1 << 16;
        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let base = E::ec_generator();
        let mut rng = thread_rng();
        let a = rng.gen_biguint(256);
        let b = rng.gen_biguint(256);
        let p_int = &base * &a;
        let q_int = &base * &b;
        let writer = generator.new_writer();
        (0..num_rows).into_par_iter().for_each(|i| {
            writer.write_ec_point(&p, &p_int, i);
            writer.write_ec_point(&q, &q_int, i);
            writer.write_ec_point(&p_pub, &p_int, i);
            writer.write_ec_point(&q_pub, &q_int, i);
            writer.write_row_instructions(&generator.air_data, i);
        });

        writer.write_global_instructions(&generator.air_data);

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);
        let public = writer.public().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public);
    }
}
