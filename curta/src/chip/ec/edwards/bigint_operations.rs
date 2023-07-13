use core::ops::{Add, Mul};

use num::BigUint;

use super::EdwardsParameters;
use crate::chip::ec::point::AffinePoint;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::utils::biguint_to_bits_le;

impl<E: EdwardsParameters> AffinePoint<E> {
    fn scalar_mul(&self, scalar: &BigUint) -> Self {
        let mut result = E::neutral();
        let mut temp = self.clone();
        let bits = biguint_to_bits_le(scalar, E::nb_scalar_bits());
        for bit in bits {
            if bit {
                result = &result + &temp;
            }
            temp = &temp + &temp;
        }
        result
    }
}

impl<E: EdwardsParameters> Add<&AffinePoint<E>> for &AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn add(self, other: &AffinePoint<E>) -> AffinePoint<E> {
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
}

impl<E: EdwardsParameters> Add<AffinePoint<E>> for AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn add(self, other: AffinePoint<E>) -> AffinePoint<E> {
        &self + &other
    }
}

impl<E: EdwardsParameters> Add<&AffinePoint<E>> for AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn add(self, other: &AffinePoint<E>) -> AffinePoint<E> {
        &self + other
    }
}

impl<E: EdwardsParameters> Mul<&BigUint> for &AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn mul(self, scalar: &BigUint) -> AffinePoint<E> {
        self.scalar_mul(scalar)
    }
}

impl<E: EdwardsParameters> Mul<BigUint> for &AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn mul(self, scalar: BigUint) -> AffinePoint<E> {
        self.scalar_mul(&scalar)
    }
}

impl<E: EdwardsParameters> Mul<BigUint> for AffinePoint<E> {
    type Output = AffinePoint<E>;

    fn mul(self, scalar: BigUint) -> AffinePoint<E> {
        self.scalar_mul(&scalar)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use num::bigint::RandBigInt;
    use num::BigUint;
    use rand::thread_rng;

    use crate::chip::ec::EllipticCurveParameters;
    use crate::chip::ec::edwards::ed25519::Ed25519;

    use super::EdwardsParameters;

    #[test]
    fn test_bigint_ed_add() {
        type E = Ed25519;
        let netural = E::neutral();
        let base = E::generator();

        assert_eq!(&base + &netural, base);
        assert_eq!(&netural + &base, base);
        assert_eq!(&netural + &netural, netural);
    }

    #[test]
    fn test_biguint_scalar_mul() {
        type E = Ed25519;
        let base = E::generator();

        let d = E::d_biguint();
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
