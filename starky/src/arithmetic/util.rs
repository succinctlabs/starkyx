use num::{BigUint, Zero};
use plonky2::field::types::Field;

pub fn bigint_into_u16_digits(x: &BigUint) -> Vec<u16> {
    x.iter_u32_digits()
        .flat_map(|x| vec![x as u16, (x >> 16) as u16])
        .collect()
}

pub fn biguint_to_16_digits<F: Field>(x: &BigUint, num_digits: usize) -> Vec<F> {
    let mut x_limbs: Vec<_> = bigint_into_u16_digits(x)
        .iter()
        .map(|xi| F::from_canonical_u16(*xi))
        .collect();
    assert!(
        x_limbs.len() <= num_digits,
        "Number too large to fit in {} digits",
        num_digits
    );
    for _ in x_limbs.len()..num_digits {
        x_limbs.push(F::ZERO);
    }
    x_limbs
}

pub fn digits_to_biguint(digits: &[u16]) -> BigUint {
    let mut x = BigUint::zero();
    for (i, &digit) in digits.iter().enumerate() {
        x += BigUint::from(digit) << (16 * i);
    }
    x
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_bigint_into_u16_digits() {
        let x = BigUint::from(0x1234567890abcdefu64);
        let x_limbs = bigint_into_u16_digits(&x);
        assert_eq!(x_limbs, vec![0xcdef, 0x90ab, 0x5678, 0x1234]);

        let mut rng = thread_rng();
        for _ in 0..100 {
            let x = rng.gen_biguint(100);
            let x_limbs = bigint_into_u16_digits(&x);

            let x_out = digits_to_biguint(&x_limbs);

            assert_eq!(x, x_out)
        }
    }
}
