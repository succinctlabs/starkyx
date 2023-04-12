use num::{BigInt, BigUint, Zero};
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;

use super::polynomial::Polynomial;

pub fn bigint_into_u16_digits(x: &BigUint, num_digits: usize) -> Vec<u16> {
    let mut x_limbs = x
        .iter_u32_digits()
        .flat_map(|x| vec![x as u16, (x >> 16) as u16])
        .collect::<Vec<_>>();
    assert!(
        x_limbs.len() <= num_digits,
        "Number too large to fit in {} digits",
        num_digits
    );
    x_limbs.resize(num_digits, 0);
    x_limbs
}

pub fn biguint_to_16_digits_field<F: Field>(x: &BigUint, num_digits: usize) -> Vec<F> {
    bigint_into_u16_digits(x, num_digits)
        .iter()
        .map(|xi| F::from_canonical_u16(*xi))
        .collect()
}

#[allow(dead_code)]
pub fn digits_to_biguint(digits: &[u16]) -> BigUint {
    let mut x = BigUint::zero();
    for (i, &digit) in digits.iter().enumerate() {
        x += BigUint::from(digit) << (16 * i);
    }
    x
}

#[allow(dead_code)]
pub fn field_limbs_to_biguint<F: RichField>(limbs: &[F]) -> BigUint {
    let mut x = BigUint::zero();
    let digits = limbs.iter().map(|x| x.to_canonical_u64());
    for (i, digit) in digits.enumerate() {
        x += BigUint::from(digit) << (16 * i);
    }
    x
}

#[inline]
pub fn split_digits<F: Field>(
    slice: &[u32],
) -> (impl Iterator<Item = F> + '_, impl Iterator<Item = F> + '_) {
    (
        slice
            .iter()
            .map(|x| *x as u16)
            .map(|x| F::from_canonical_u16(x)),
        slice
            .iter()
            .map(|x| (*x >> 16) as u16)
            .map(|x| F::from_canonical_u16(x)),
    )
}

#[inline]
pub fn extract_witness_and_shift(vanishing_poly: &Polynomial<i64>, offset: u32) -> Vec<u32> {
    //sanity check
    let eval_vanishing = vanishing_poly
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, x)| BigInt::from(2u32).pow(16 * i as u32) * x)
        .sum::<BigInt>();
    debug_assert_eq!(eval_vanishing, BigInt::from(0));

    // extract witness from vanishing polynomial
    let limb = 2u32.pow(16) as i64;
    let witness_poly = vanishing_poly.root_quotient(limb);
    assert_eq!(witness_poly.degree(), vanishing_poly.degree() - 1);

    for c in witness_poly.as_slice().iter() {
        debug_assert!(c.abs() < offset as i64);
    }

    // Sanity check
    debug_assert_eq!(
        (&witness_poly * &(Polynomial::<i64>::new_from_slice(&[-limb, 1]))).as_slice(),
        vanishing_poly.as_slice()
    );

    // Shifting the witness polynomial to make it positive
    witness_poly
        .coefficients()
        .iter()
        .map(|x| x + offset as i64)
        .map(|x| u32::try_from(x).unwrap())
        .collect()
}

#[inline]
pub fn to_field_iter<F: Field>(polynomial: &Polynomial<i64>) -> impl Iterator<Item = F> + '_ {
    polynomial
        .as_slice()
        .iter()
        .map(|x| F::from_canonical_u32(*x as u32))
}

pub fn biguint_to_bits_le(integer: &BigUint, num_bits: usize) -> Vec<bool> {
    let byte_vec = integer.to_bytes_le();
    let mut bits = Vec::new();
    for byte in byte_vec {
        for i in 0..8 {
            bits.push(byte & (1 << i) != 0);
        }
    }
    debug_assert!(
        bits.len() <= num_bits,
        "Number too large to fit in {} digits",
        num_bits
    );
    bits.resize(num_bits, false);
    bits
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_bigint_into_u16_digits() {
        let x = BigUint::from(0x1234567890abcdefu64);
        let x_limbs = bigint_into_u16_digits(&x, 4);
        assert_eq!(x_limbs, vec![0xcdef, 0x90ab, 0x5678, 0x1234]);

        let mut rng = thread_rng();
        for _ in 0..100 {
            let x = rng.gen_biguint(256);
            let x_limbs = bigint_into_u16_digits(&x, 16);

            let x_out = digits_to_biguint(&x_limbs);

            assert_eq!(x, x_out)
        }
    }

    #[test]
    fn test_into_bits_le() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let x = rng.gen_biguint(256);
            let bits = biguint_to_bits_le(&x, 256);
            let mut x_out = BigUint::from(0u32);
            for (i, bit) in bits.iter().enumerate() {
                if *bit {
                    x_out += BigUint::from(1u32) << i;
                }
            }
            assert_eq!(x, x_out);
        }
    }
}
