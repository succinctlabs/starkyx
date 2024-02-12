use crate::math::field::{Field, PrimeField64};

#[inline]
pub fn u32_to_le_field_bytes<F: Field>(value: u32) -> [F; 4] {
    value.to_le_bytes().map(F::from_canonical_u8)
}

#[inline]
pub fn u32_from_le_field_bytes<F: PrimeField64>(bytes: &[F; 4]) -> u32 {
    u32::from_le_bytes(bytes.map(|x| x.as_canonical_u64() as u8))
}

#[inline]
pub fn u64_to_le_field_bytes<F: Field>(value: u64) -> [F; 8] {
    value.to_le_bytes().map(F::from_canonical_u8)
}

#[inline]
pub fn u64_from_le_field_bytes<F: PrimeField64>(bytes: &[F; 8]) -> u64 {
    u64::from_le_bytes(bytes.map(|x| x.as_canonical_u64() as u8))
}
