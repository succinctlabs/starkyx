use itertools::Itertools;

use super::ed25519::{Ed25519BaseField, Ed25519Parameters, FromEd25519FieldInstruction};
use super::{EdwardsCurve, EdwardsParameters};
use crate::chip::builder::AirBuilder;
use crate::chip::ec::point::{AffinePointRegister, CompressedPointRegister};
use crate::chip::field::parameters::FieldParameters;
use crate::chip::field::register::FieldRegister;
use crate::chip::AirParameters;
use crate::math::field::Field;
use crate::polynomial::Polynomial;

impl<L: AirParameters> AirBuilder<L> {
    pub fn ed_decompress(
        &mut self,
        compressed_p: &CompressedPointRegister,
    ) -> AffinePointRegister<EdwardsCurve<Ed25519Parameters>>
    where
        L::Instruction: FromEd25519FieldInstruction,
    {
        // Ed25519 Elliptic Curve Decompress Formula
        //
        // This function uses a similar logic as this function:
        // https://github.com/succinctlabs/curve25519-dalek/blob/e2d1bd10d6d772af07cac5c8161cd7655016af6d/curve25519-dalek/src/edwards.rs#L187
        let mut one_limbs = [0; Ed25519BaseField::NB_LIMBS];
        one_limbs[0] = 1;
        let one_p = Polynomial::<L::Field>::from_coefficients(
            one_limbs.map(L::Field::from_canonical_u16).to_vec(),
        );
        let one = self.constant::<FieldRegister<Ed25519BaseField>>(&one_p);

        let d_p = Polynomial::<L::Field>::from_coefficients(
            Ed25519Parameters::D[0..Ed25519BaseField::NB_LIMBS]
                .iter()
                .map(|x| L::Field::from_canonical_u16(*x))
                .collect_vec(),
        );

        let d = self.constant::<FieldRegister<Ed25519BaseField>>(&d_p);

        let zero_limbs = [0; Ed25519BaseField::NB_LIMBS];
        let zero_p = Polynomial::<L::Field>::from_coefficients(
            zero_limbs.map(L::Field::from_canonical_u16).to_vec(),
        );
        let zero = self.constant::<FieldRegister<Ed25519BaseField>>(&zero_p);

        let yy = self.fp_mul::<Ed25519BaseField>(&compressed_p.y, &compressed_p.y);
        let u = self.fp_sub::<Ed25519BaseField>(&yy, &one);
        let dyy = self.fp_mul::<Ed25519BaseField>(&d, &compressed_p.y);
        let v = self.fp_add::<Ed25519BaseField>(&one, &dyy);
        let u_div_v = self.fp_div::<Ed25519BaseField>(&u, &v);

        let mut x = self.fp_sqrt(&u_div_v);
        let neg_x = self.fp_sub::<Ed25519BaseField>(&zero, &x);
        x = self.select(&compressed_p.sign, &x, &neg_x);

        AffinePointRegister::<EdwardsCurve<Ed25519Parameters>>::new(x, compressed_p.y)
    }
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use curve25519_dalek::edwards::CompressedEdwardsY;
    use num::BigUint;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::edwards::ed25519::Ed25519FpInstruction;
    use crate::chip::ec::gadget::{
        CompressedPointGadget, CompressedPointWriter, EllipticCurveWriter,
    };
    use crate::chip::ec::point::AffinePoint;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    pub struct Ed25519DecompressTest;

    impl AirParameters for Ed25519DecompressTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 784;
        const NUM_FREE_COLUMNS: usize = 3;
        const EXTENDED_COLUMNS: usize = 1185;
        type Instruction = Ed25519FpInstruction;
    }

    #[test]
    fn decompress_point() {
        let compressed_p_hex = "092005a6f7a58a98df5f9b8d186b9877f12b603aa06c7debf0f610d5a49f9ed7";
        let compressed_p_bytes = hex::decode(compressed_p_hex).unwrap();
        let compressed_p = CompressedEdwardsY(compressed_p_bytes.try_into().unwrap());

        let point = compressed_p.decompress().unwrap();
        println!(
            "x is {:?}",
            BigUint::from_bytes_le(&point.get_x().as_bytes())
        );
        println!(
            "y is {:?}",
            BigUint::from_bytes_le(&point.get_y().as_bytes())
        );
    }

    #[test]
    fn test_ed25519_decompress() {
        type L = Ed25519DecompressTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let compressed_p_reg = builder.alloc_ec_compressed_point();
        let _affine_p_reg = builder.ed_decompress(&compressed_p_reg);

        let num_rows = 1 << 16;
        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let compressed_p_hex = "092005a6f7a58a98df5f9b8d186b9877f12b603aa06c7debf0f610d5a49f9ed7";
        let compressed_p_bytes = hex::decode(compressed_p_hex).unwrap();
        let compressed_p = CompressedEdwardsY(compressed_p_bytes.try_into().unwrap());

        let affine_p_x = BigUint::from_str(
            "46498302853928694666678954892487633213173616891947474062122140763021808627271",
        )
        .unwrap();
        let affine_p_y = BigUint::from_str(
            "39631481484518050587569957685312118971016858466473658974116614941450822819849",
        )
        .unwrap();
        let affine_p = AffinePoint::<EdwardsCurve<Ed25519Parameters>>::new(affine_p_x, affine_p_y);

        let writer = generator.new_writer();
        (0..num_rows).into_par_iter().for_each(|i| {
            writer.write_ec_compressed_point(&compressed_p_reg, &compressed_p, i);
            //writer.write_ec_point(&affine_p_reg, &affine_p, i);
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
