use num::BigUint;

use super::{SWCurve, WeierstrassParameters};
use crate::chip::ec::point::AffinePoint;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::utils::biguint_to_bits_le;

impl<E: WeierstrassParameters> AffinePoint<SWCurve<E>> {
    pub fn sw_scalar_mul(&self, scalar: &BigUint) -> Self {
        let mut result: Option<AffinePoint<SWCurve<E>>> = None;
        let mut temp = self.clone();
        let bits = biguint_to_bits_le(scalar, E::nb_scalar_bits());
        for bit in bits {
            if bit {
                result = result.map(|r| r.sw_add(&temp)).or(Some(temp.clone()));
            }
            temp = temp.sw_double();
        }
        result.unwrap()
    }
}

impl<E: WeierstrassParameters> AffinePoint<SWCurve<E>> {
    pub fn sw_add(&self, other: &AffinePoint<SWCurve<E>>) -> AffinePoint<SWCurve<E>> {
        let p = E::BaseField::modulus();
        let slope_numerator = (&p + &other.y - &self.y) % &p;
        let slope_denominator = (&p + &other.x - &self.x) % &p;
        let slope_denom_inverse = slope_denominator.modpow(&(&p - 2u32), &p);
        let slope = (slope_numerator * &slope_denom_inverse) % &p;

        let x_3n = (&slope * &slope + &p + &p - &self.x - &other.x) % &p;
        let y_3n = (&slope * &(&p + &self.x - &x_3n) + &p - &self.y) % &p;

        AffinePoint::new(x_3n, y_3n)
    }

    pub fn sw_double(&self) -> AffinePoint<SWCurve<E>> {
        let p = E::BaseField::modulus();
        let a = E::a_int();
        let slope_numerator = (&a + &(&self.x * &self.x) * 3u32) % &p;

        let slope_denominator = (&self.y * 2u32) % &p;
        let slope_denom_inverse = slope_denominator.modpow(&(&p - 2u32), &p);
        let slope = (slope_numerator * &slope_denom_inverse) % &p;

        let x_3n = (&slope * &slope + &p + &p - &self.x - &self.x) % &p;

        let y_3n = (&slope * &(&p + &self.x - &x_3n) + &p - &self.y) % &p;

        AffinePoint::new(x_3n, y_3n)
    }
}

#[cfg(test)]
mod tests {

    use num::bigint::RandBigInt;
    use rand::thread_rng;

    use crate::chip::ec::weierstrass::bn254::Bn254;

    #[test]
    fn test_weierstrass_biguint_scalar_mul() {
        type E = Bn254;
        let base = E::generator();

        let mut rng = thread_rng();
        for _ in 0..10 {
            let x = rng.gen_biguint(24);
            let y = rng.gen_biguint(25);

            let x_base = base.sw_scalar_mul(&x);
            let y_x_base = x_base.sw_scalar_mul(&y);
            let xy = &x * &y;
            let xy_base = base.sw_scalar_mul(&xy);
            assert_eq!(y_x_base, xy_base);
        }
    }
}
