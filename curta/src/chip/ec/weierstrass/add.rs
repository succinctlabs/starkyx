//! Elliptic curve addition gadget

use super::projective::SWProjectivePointRegister;
use super::WeierstrassParameter;
use crate::chip::builder::AirBuilder;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::AirParameters;

/// Gadgets for Weierstrass curve projective point addition
/// Formula: [http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-1998-cmo-2]
#[derive(Debug, Clone)]
pub struct SWAddGadget<E: WeierstrassParameter> {
    pub p: SWProjectivePointRegister<E>,
    pub q: SWProjectivePointRegister<E>,
    pub result: SWProjectivePointRegister<E>,
    // z1z1_ins: FpMulInstruction<E::BaseField>,
    // z1z1z1_ins: FpMulInstruction<E::BaseField>,
    // z2z2_ins: FpMulInstruction<E::BaseField>,
    // z2z2z2_ins: FpMulInstruction<E::BaseField>,
    // u1_ins: FpMulInstruction<E::BaseField>,
    // u2_ins: FpMulInstruction<E::BaseField>,
    // s1_ins: FpMulInstruction<E::BaseField>,
    // s2_ins: FpMulInstruction<E::BaseField>,
    // hh_ins: FpMulInstruction<E::BaseField>,
    // hhh_ins: FpMulInstruction<E::BaseField>,
    // v_ins: FpMulInstruction<E::BaseField>,
    // rr_ins: FpMulInstruction<E::BaseField>,
    // r_v_sub_x3_ins: FpMulInstruction<E::BaseField>,
    // s1_hhh_ins: FpMulInstruction<E::BaseField>,
    // z1z2_ins: FpMulInstruction<E::BaseField>,
    // z3_ins: FpMulInstruction<E::BaseField>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn sw_projective_add<E: WeierstrassParameter>(
        &mut self,
        p: &SWProjectivePointRegister<E>,
        q: &SWProjectivePointRegister<E>,
    ) -> SWAddGadget<E>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        // Addition formula for Weierstrass curve projective points
        //
        // Sum of (x1, y1, z1) and (x2, y2, z2) is (x3, y3, z3) where
        //  z1z1 = z1 * z1
        //  z1z1z1 = z1z1 * z1
        //  z2z2 = z2 * z2
        //  z2z2z2 = z2z2 * z2
        //  u1 = x1 * z2z2
        //  u2 = x2 * z1z1
        //  s1 = y1 * z2z2z2
        //  s2 = y2 * z1z1z1
        //  h = u2 - u1
        //  hh = h * h
        //  hhh = hh * h
        //  r = s2 - s1
        //  v = u1 * hh
        //  x3 = r * r - hhh - 2 * v
        //  y3 = r * (v - x3) - s1 * hhh
        //  z3 = z1 * z2 * h
        //
        // Reference: http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-1998-cmo-2

        let x1 = p.x;
        let x2 = q.x;
        let y1 = p.y;
        let y2 = q.y;
        let z1 = p.z;
        let z2 = q.z;

        let z1z1_ins = self.fp_mul(&z1, &z1);
        let z1z1z1_ins = self.fp_mul(&z1z1_ins.result, &z1);
        let z2z2_ins = self.fp_mul(&z2, &z2);
        let z2z2z2_ins = self.fp_mul(&z2z2_ins.result, &z2);

        let u1_ins = self.fp_mul(&x1, &z2z2_ins.result);
        let u2_ins = self.fp_mul(&x2, &z1z1_ins.result);
        let s1_ins = self.fp_mul(&y1, &z2z2z2_ins.result);
        let s2_ins = self.fp_mul(&y2, &z1z1z1_ins.result);

        let h_ins = self.fp_sub(&u2_ins.result, &u1_ins.result);
        let hh_ins = self.fp_mul(&h_ins, &h_ins);
        let hhh_ins = self.fp_mul(&hh_ins.result, &h_ins);

        let r_ins = self.fp_sub(&s2_ins.result, &s1_ins.result);
        let v_ins = self.fp_mul(&u1_ins.result, &hh_ins.result);

        // Calculate x3 = r * r - hhh - 2 * v
        let rr_ins = self.fp_mul(&r_ins, &r_ins);
        let rr_sub_hhh_ins = self.fp_sub(&rr_ins.result, &hhh_ins.result);
        let v_plus_v_ins = self.fp_add(&v_ins.result, &v_ins.result);
        let x3_ins = self.fp_sub(&rr_sub_hhh_ins, &v_plus_v_ins);

        // Calculate y3 = r * (v - x3) - s1 * hhh
        let v_sub_x3_ins = self.fp_sub(&v_ins.result, &x3_ins);
        let r_v_sub_x3_ins = self.fp_mul(&r_ins, &v_sub_x3_ins);
        let s1_hhh_ins = self.fp_mul(&s1_ins.result, &hhh_ins.result);
        let y3_ins = self.fp_sub(&r_v_sub_x3_ins.result, &s1_hhh_ins.result);

        // Calculate z3 = z1 * z2 * h
        let z1z2_ins = self.fp_mul(&z1, &z2);
        let z3_ins = self.fp_mul(&z1z2_ins.result, &h_ins);

        let result = SWProjectivePointRegister::new(x3_ins, y3_ins, z3_ins.result);

        SWAddGadget {
            p: *p,
            q: *q,
            result,
            // z1z1_ins,
            // z1z1z1_ins,
            // z2z2_ins,
            // z2z2z2_ins,
            // u1_ins,
            // u2_ins,
            // s1_ins,
            // s2_ins,
            // hh_ins,
            // hhh_ins,
            // v_ins,
            // rr_ins,
            // r_v_sub_x3_ins,
            // s1_hhh_ins,
            // z1z2_ins,
            // z3_ins,
        }
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::weierstrass::bn254::{Bn254, Bn254BaseField};
    use crate::chip::ec::weierstrass::projective::{
        SWEllipticCurveWriter, SWProjectiveEllipticCurveGadget,
    };
    use crate::chip::field::instruction::FpInstruction;

    #[derive(Clone, Debug, Copy)]
    pub struct SWAddTest;

    impl AirParameters for SWAddTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 2212;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 3327;
        type Instruction = FpInstruction<Bn254BaseField>;

        fn num_rows_bits() -> usize {
            16
        }
    }

    #[test]
    fn test_bn254_add() {
        type L = SWAddTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Bn254;

        let mut builder = AirBuilder::<L>::new();

        let p = builder.alloc_swec_point();
        let q = builder.alloc_swec_point();

        let _gadget = builder.sw_projective_add::<E>(&p, &q);

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
            writer.write_swec_point(&p, &p_int, i);
            writer.write_swec_point(&q, &q_int, i);
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
