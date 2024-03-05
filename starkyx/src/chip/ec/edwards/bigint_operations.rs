use super::{EdwardsCurve, EdwardsParameters};
use crate::chip::ec::point::AffinePoint;
use crate::chip::field::parameters::FieldParameters;

impl<E: EdwardsParameters> AffinePoint<EdwardsCurve<E>> {
    pub(crate) fn ed_add(
        &self,
        other: &AffinePoint<EdwardsCurve<E>>,
    ) -> AffinePoint<EdwardsCurve<E>> {
        let p = E::BaseField::modulus();
        let x_3n = (&self.x * &other.y + &self.y * &other.x) % &p;
        let y_3n = (&self.y * &other.y + &self.x * &other.x) % &p;

        let all_xy = (&self.x * &self.y * &other.x * &other.y) % &p;
        let d = E::d_biguint();
        let dxy = (d * &all_xy) % &p;
        let den_x = ((1u32 + &dxy) % &p).modpow(&(&p - 2u32), &p);
        let den_y = ((1u32 + &p - &dxy) % &p).modpow(&(&p - 2u32), &p);

        let x_3 = (&x_3n * &den_x) % &p;
        let y_3 = (&y_3n * &den_y) % &p;

        AffinePoint::new(x_3, y_3)
    }

    pub(crate) fn ed_double(&self) -> AffinePoint<EdwardsCurve<E>> {
        self.ed_add(self)
    }
}

#[cfg(test)]
mod tests {

    use num::bigint::RandBigInt;
    use num::BigUint;
    use rand::thread_rng;

    use super::*;
    use crate::chip::ec::edwards::ed25519::params::{Ed25519, Ed25519Parameters};
    use crate::chip::ec::{EllipticCurve, EllipticCurveParameters};

    #[test]
    fn test_bigint_ed_add() {
        type E = Ed25519;
        let netural = E::neutral();
        let base = E::ec_generator();

        assert_eq!(&base + &netural, base);
        assert_eq!(&netural + &base, base);
        assert_eq!(&netural + &netural, netural);
    }

    #[test]
    fn test_biguint_scalar_mul() {
        type E = Ed25519;
        let base = E::ec_generator();

        let d = Ed25519Parameters::d_biguint();
        let p = <E as EllipticCurveParameters>::BaseField::modulus();
        assert_eq!((d * 121666u32) % &p, (&p - 121665u32) % &p);

        let mut rng = thread_rng();
        for _ in 0..10 {
            let x = rng.gen_biguint(24);
            let y = rng.gen_biguint(25);

            let x_base = &base * &x;
            let y_x_base = &x_base * &y;
            let xy = &x * &y;
            let xy_base = &base * &xy;
            assert_eq!(y_x_base, xy_base);
        }

        let order = BigUint::from(2u32).pow(252)
            + BigUint::from(27742317777372353535851937790883648493u128);
        assert_eq!(base, &base + &(&base * &order));
    }
}
