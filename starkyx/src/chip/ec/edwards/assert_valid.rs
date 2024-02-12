use super::{EdwardsCurve, EdwardsParameters};
use crate::chip::builder::AirBuilder;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::add::FpAddInstruction;
use crate::chip::field::mul::FpMulInstruction;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::AirParameters;
use crate::math::field::Field;
use crate::polynomial::Polynomial;

impl<L: AirParameters> AirBuilder<L> {
    pub fn ed_assert_valid<E: EdwardsParameters>(
        &mut self,
        p: &AffinePointRegister<EdwardsCurve<E>>,
    ) where
        L::Instruction: From<FpMulInstruction<E::BaseField>> + From<FpAddInstruction<E::BaseField>>,
    {
        // Ed25519 Elliptic Curve Assert Valid
        //
        // Equation: a * x ** 2 + y ** 2 = 1 + d * x ** 2 * y ** 2
        // a is -1, so the above equation can be rewritten as
        // y ** 2 = 1 + d * x ** 2 * y ** 2 + x ** 2
        let num_limbs: usize = E::BaseField::NB_LIMBS;
        let mut one_limbs = vec![0u16; num_limbs];
        one_limbs[0] = 1;
        let one_p = Polynomial::<L::Field>::from_coefficients(
            one_limbs
                .iter()
                .map(|x| L::Field::from_canonical_u16(*x))
                .collect::<Vec<_>>(),
        );
        let one = self.constant(&one_p);

        let d_p = Polynomial::<L::Field>::from_coefficients(
            E::D[0..num_limbs]
                .iter()
                .map(|x| L::Field::from_canonical_u16(*x))
                .collect::<Vec<_>>(),
        );

        let d = self.constant(&d_p);

        let y_squared = self.fp_mul(&p.y, &p.y);
        let x_squared = self.fp_mul(&p.x, &p.x);
        let x_squared_times_y_squared = self.fp_mul(&x_squared, &y_squared);
        let d_x_squared_times_y_squared = self.fp_mul(&d, &x_squared_times_y_squared);
        let d_x_squared_times_y_squared_plus_x_sqaured =
            self.fp_add(&d_x_squared_times_y_squared, &x_squared);
        let rhs = self.fp_add(&one, &d_x_squared_times_y_squared_plus_x_sqaured);

        self.assert_equal(&y_squared, &rhs);
    }
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use num::BigUint;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::edwards::ed25519::params::{Ed25519BaseField, Ed25519Parameters};
    use crate::chip::ec::gadget::{EllipticCurveGadget, EllipticCurveWriter};
    use crate::chip::ec::point::AffinePoint;
    use crate::chip::field::instruction::FpInstruction;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    pub struct Ed25519AssertValidTest;

    impl AirParameters for Ed25519AssertValidTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 584;
        const NUM_FREE_COLUMNS: usize = 2;
        const EXTENDED_COLUMNS: usize = 885;
        type Instruction = FpInstruction<Ed25519BaseField>;
    }

    #[test]
    fn test_ed25519_assert_valid() {
        type L = Ed25519AssertValidTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Ed25519Parameters;

        let mut builder = AirBuilder::<L>::new();

        let p = builder.alloc_ec_point();
        builder.ed_assert_valid::<E>(&p);

        let num_rows = 1 << 16;
        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let p_x = BigUint::from_str(
            "46498302853928694666678954892487633213173616891947474062122140763021808627271",
        )
        .unwrap();
        let p_y = BigUint::from_str(
            "39631481484518050587569957685312118971016858466473658974116614941450822819849",
        )
        .unwrap();
        let affine_p = AffinePoint::<EdwardsCurve<E>>::new(p_x, p_y);

        let writer = generator.new_writer();
        writer.write_global_instructions(&generator.air_data);

        (0..num_rows).into_par_iter().for_each(|i| {
            writer.write_ec_point(&p, &affine_p, i);
            writer.write_row_instructions(&generator.air_data, i);
        });

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);
        let public = writer.public().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public);
    }

    #[test]
    #[should_panic]
    fn test_ed25519_assert_not_valid() {
        type L = Ed25519AssertValidTest;
        type SC = PoseidonGoldilocksStarkConfig;
        type E = Ed25519Parameters;

        let mut builder = AirBuilder::<L>::new();

        let p = builder.alloc_ec_point();
        builder.ed_assert_valid::<E>(&p);

        let num_rows = 1 << 16;
        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let p_x = BigUint::from_str(
            "46498302853928694666678954892487633213173616891947474062122140763021808627271",
        )
        .unwrap();
        let p_y = BigUint::from_str(
            "39631481484518050587569957685312118971016858466473658974116614941450822819848",
        )
        .unwrap();
        let affine_p = AffinePoint::<EdwardsCurve<E>>::new(p_x, p_y);

        let writer = generator.new_writer();
        writer.write_global_instructions(&generator.air_data);

        (0..num_rows).into_par_iter().for_each(|i| {
            writer.write_ec_point(&p, &affine_p, i);
            writer.write_row_instructions(&generator.air_data, i);
        });

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);
        let public = writer.public().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public);
    }
}
