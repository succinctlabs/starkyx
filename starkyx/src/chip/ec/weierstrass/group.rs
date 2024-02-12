use super::{SWCurve, WeierstrassParameters};
use crate::chip::builder::AirBuilder;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::register::FieldRegister;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    /// Given two points `p` and `q` and the slope of the intersection line, compute the addition.
    pub(crate) fn sw_add_with_slope<E: WeierstrassParameters>(
        &mut self,
        p: &AffinePointRegister<SWCurve<E>>,
        q: &AffinePointRegister<SWCurve<E>>,
        slope: &FieldRegister<E::BaseField>,
    ) -> AffinePointRegister<SWCurve<E>>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        let (x_1, y_1) = (p.x, p.y);
        let x_2 = q.x;

        let slope_squared = self.fp_mul(slope, slope);

        let mut x_3 = self.fp_sub(&slope_squared, &x_1);
        x_3 = self.fp_sub(&x_3, &x_2);

        let mut y_3 = self.fp_sub(&x_1, &x_3);
        y_3 = self.fp_mul(slope, &y_3);
        y_3 = self.fp_sub(&y_3, &y_1);

        AffinePointRegister::<SWCurve<E>> { x: x_3, y: y_3 }
    }

    /// Add two different points `p` and `q` on a short Weierstrass curve.
    pub fn sw_add<E: WeierstrassParameters>(
        &mut self,
        p: &AffinePointRegister<SWCurve<E>>,
        q: &AffinePointRegister<SWCurve<E>>,
    ) -> AffinePointRegister<SWCurve<E>>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        let slope = self.sw_slope_different(p, q);
        self.sw_add_with_slope(p, q, &slope)
    }

    /// Doubles a point `p` on a short Weierstrass curve.
    pub fn sw_double<E: WeierstrassParameters>(
        &mut self,
        p: &AffinePointRegister<SWCurve<E>>,
        a: &FieldRegister<E::BaseField>,
        three: &FieldRegister<E::BaseField>,
    ) -> AffinePointRegister<SWCurve<E>>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        let slope = self.sw_tangent(p, a, three);
        self.sw_add_with_slope(p, p, &slope)
    }
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::gadget::{EllipticCurveGadget, EllipticCurveWriter};
    use crate::chip::ec::weierstrass::bn254::{Bn254, Bn254BaseField};
    use crate::chip::field::instruction::FpInstruction;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    pub struct Ed25519AddTest;

    impl AirParameters for Ed25519AddTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 1152;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 1737;
        type Instruction = FpInstruction<Bn254BaseField>;
    }

    #[test]
    fn test_bn254_add() {
        type L = Ed25519AddTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Bn254;

        let mut builder = AirBuilder::<L>::new();

        let p = builder.alloc_ec_point();
        let q = builder.alloc_ec_point();

        let _res = builder.ec_add(&p, &q);

        let num_rows = 1 << 16;
        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let base = E::generator();
        let mut rng = thread_rng();
        let a = rng.gen_biguint(256);
        let b = rng.gen_biguint(256);
        let p_int = base.sw_scalar_mul(&a);
        let q_int = base.sw_scalar_mul(&b);
        let writer = generator.new_writer();
        (0..num_rows).for_each(|i| {
            writer.write_ec_point(&p, &p_int, i);
            writer.write_ec_point(&q, &q_int, i);
            writer.write_row_instructions(&generator.air_data, i);
        });

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        let public_inputs = writer.0.public.read().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }

    #[test]
    fn test_bn254_double() {
        type L = Ed25519AddTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Bn254;

        let mut builder = AirBuilder::<L>::new();

        let p = builder.alloc_ec_point();

        let _res = builder.ec_double(&p);

        let num_rows = 1 << 16;
        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let base = E::generator();
        let p_int = &base;
        let writer = generator.new_writer();
        writer.write_global_instructions(&generator.air_data);
        (0..num_rows).for_each(|i| {
            writer.write_ec_point(&p, p_int, i);
            writer.write_row_instructions(&generator.air_data, i);
        });

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);

        let public_inputs = writer.0.public.read().unwrap().clone();
        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public_inputs);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public_inputs);
    }
}
