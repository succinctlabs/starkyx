//! Gadgets for SW elliptic curve projective points
use core::marker::PhantomData;
use core::ops::{Add, Mul};

use num::{BigUint, Zero};
use plonky2::field::types::PrimeField64;

use super::WeierstrassParameter;
use crate::chip::builder::AirBuilder;
use crate::chip::ec::EllipticCurveParameters;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::field::register::FieldRegister;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::chip::utils::{biguint_to_bits_le, field_limbs_to_biguint};
use crate::chip::AirParameters;
use crate::polynomial::to_u16_le_limbs_polynomial;

#[derive(Debug, Clone, Eq)]
/// short Weierstrass projective point
pub struct SWProjectivePoint<E: EllipticCurveParameters> {
    /// x coordinate
    pub x: BigUint,
    /// y coordinate
    pub y: BigUint,
    /// z coordinate
    pub z: BigUint,
    _marker: std::marker::PhantomData<E>,
}

impl<E: EllipticCurveParameters> SWProjectivePoint<E> {
    /// Create a new SW affine point
    pub fn new(x: BigUint, y: BigUint, z: BigUint) -> Self {
        Self {
            x,
            y,
            z,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: EllipticCurveParameters> PartialEq for SWProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        if self.z.is_zero() {
            return other.z.is_zero();
        }
        let modulus = &E::BaseField::modulus();
        let zz1 = (&self.z * &self.z) % modulus;
        let zzz1 = (&zz1 * &self.z) % modulus;
        let zz2 = (&other.z * &other.z) % modulus;
        let zzz2 = (&zz2 * &other.z) % modulus;
        ((&self.x * zz2 % modulus) == (&other.x * zz1 % modulus))
            && ((&self.y * zzz2 % modulus) == (&other.y * zzz1 % modulus))
    }
}

impl<E: WeierstrassParameter> SWProjectivePoint<E> {
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

impl<E: WeierstrassParameter> Add<&SWProjectivePoint<E>> for &SWProjectivePoint<E> {
    type Output = SWProjectivePoint<E>;

    fn add(self, other: &SWProjectivePoint<E>) -> SWProjectivePoint<E> {
        let modulus = &E::BaseField::modulus();
        let (x1, y1, z1) = (&self.x, &self.y, &self.z);
        let (x2, y2, z2) = (&other.x, &other.y, &other.z);
        if self != other {
            // Addition formula for Weierstrass curve projective points
            // Reference: http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-1998-cmo-2
            let z1z1 = z1 * z1 % modulus;
            let z1z1z1 = &z1z1 * z1 % modulus;
            let z2z2 = z2 * z2 % modulus;
            let z2z2z2 = &z2z2 * z2 % modulus;
            let (u1, u2) = (x1 * &z2z2 % modulus, x2 * &z1z1 % modulus);
            let (s1, s2) = (y1 * &z2z2z2 % modulus, y2 * &z1z1z1 % modulus);
            let h = (&u2 + modulus - &u1) % modulus;
            let hh = &h * &h % modulus;
            let hhh = &hh * &h % modulus;
            let r = (&s2 + modulus - &s1) % modulus;
            let v = &u1 * &hh % modulus;
            let x3 = (&r * &r + modulus * 3u32 - &hhh - &v - &v) % modulus;
            let y3 =
                (&r * (&v + modulus - &x3) % modulus + modulus - &s1 * &hhh % modulus) % modulus;
            let z3 = z1 * z2 * h % modulus;
            SWProjectivePoint {
                x: x3,
                y: y3,
                z: z3,
                _marker: PhantomData,
            }
        } else {
            // Doubling formula for Weierstrass curve projective points
            // Reference: http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2007-bl
            let xx = x1 * x1 % modulus;
            let yy = y1 * y1 % modulus;
            let yyyy = &yy * &yy % modulus;
            let zz = z1 * z1 % modulus;
            let s = (x1 + &yy) % modulus;
            let s = ((&s * &s) % modulus + modulus - &xx + modulus - &yyyy) * 2u32 % modulus;
            let m = (xx * 3u32 + (&zz * &zz % modulus * E::a_biguint())) % modulus;
            let x3 = (&m * &m % modulus + modulus - &s + modulus - &s) % modulus;
            let y3 =
                ((&m * (s + modulus - &x3) % modulus) + modulus * 8u32 - yyyy * 8u32) % modulus;
            let z3 = ((y1 + z1) * (y1 + z1) % modulus + modulus - yy + modulus - zz) % modulus;
            SWProjectivePoint {
                x: x3,
                y: y3,
                z: z3,
                _marker: PhantomData,
            }
        }
    }
}

impl<E: WeierstrassParameter> Add<SWProjectivePoint<E>> for SWProjectivePoint<E> {
    type Output = SWProjectivePoint<E>;

    fn add(self, other: SWProjectivePoint<E>) -> SWProjectivePoint<E> {
        &self + &other
    }
}

impl<E: WeierstrassParameter> Add<&SWProjectivePoint<E>> for SWProjectivePoint<E> {
    type Output = SWProjectivePoint<E>;

    fn add(self, other: &SWProjectivePoint<E>) -> SWProjectivePoint<E> {
        &self + other
    }
}

impl<E: WeierstrassParameter> Mul<&BigUint> for &SWProjectivePoint<E> {
    type Output = SWProjectivePoint<E>;

    fn mul(self, scalar: &BigUint) -> SWProjectivePoint<E> {
        self.scalar_mul(scalar)
    }
}

impl<E: WeierstrassParameter> Mul<BigUint> for &SWProjectivePoint<E> {
    type Output = SWProjectivePoint<E>;

    fn mul(self, scalar: BigUint) -> SWProjectivePoint<E> {
        self.scalar_mul(&scalar)
    }
}

impl<E: WeierstrassParameter> Mul<BigUint> for SWProjectivePoint<E> {
    type Output = SWProjectivePoint<E>;

    fn mul(self, scalar: BigUint) -> SWProjectivePoint<E> {
        self.scalar_mul(&scalar)
    }
}

#[derive(Debug, Clone, Copy)]
/// Register for short Weierstrass projective point
pub struct SWProjectivePointRegister<E: EllipticCurveParameters> {
    /// x coordinate
    pub x: FieldRegister<E::BaseField>,
    /// y coordinate
    pub y: FieldRegister<E::BaseField>,
    /// z coordinate
    pub z: FieldRegister<E::BaseField>,
}

impl<E: EllipticCurveParameters> SWProjectivePointRegister<E> {
    /// Create a new SW projective point register
    pub fn new(
        x: FieldRegister<E::BaseField>,
        y: FieldRegister<E::BaseField>,
        z: FieldRegister<E::BaseField>,
    ) -> Self {
        Self { x, y, z }
    }
}

/// Gadget for short Weierstrass elliptic curve
pub trait SWProjectiveEllipticCurveGadget<E: EllipticCurveParameters> {
    /// Allocates registers for a SW Projective elliptic curve point without
    /// range-checking.
    fn alloc_unchecked_sw_point(&mut self) -> SWProjectivePointRegister<E>;

    /// Allocates local registers for a SW Projective point
    fn alloc_local_sw_point(&mut self) -> SWProjectivePointRegister<E>;

    /// Allocates next registers for a SW Projective point
    fn alloc_next_sw_point(&mut self) -> SWProjectivePointRegister<E>;

    /// Allocates registers for a SW Projective point
    fn alloc_sw_point(&mut self) -> SWProjectivePointRegister<E> {
        self.alloc_local_sw_point()
    }

    /// Allocates global registers for a SW Projective point
    fn alloc_global_sw_point(&mut self) -> SWProjectivePointRegister<E>;

    /// Allocates public registers for a SW Projective point
    fn alloc_public_sw_point(&mut self) -> SWProjectivePointRegister<E>;
}

/// I/O for SW elliptic curve registers
pub trait SWEllipticCurveWriter<E: EllipticCurveParameters> {
    /// Read SW elliptic curve point from registers
    fn read_sw_point(
        &self,
        data: &SWProjectivePointRegister<E>,
        row_index: usize,
    ) -> SWProjectivePoint<E>;

    /// Write SW elliptic curve point into registers
    fn write_sw_point(
        &self,
        data: &SWProjectivePointRegister<E>,
        value: &SWProjectivePoint<E>,
        row_index: usize,
    );
}

impl<L: AirParameters, E: EllipticCurveParameters> SWProjectiveEllipticCurveGadget<E>
    for AirBuilder<L>
{
    /// Allocates registers for a SW Projective elliptic curve point without
    /// range-checking.
    fn alloc_unchecked_sw_point(&mut self) -> SWProjectivePointRegister<E> {
        let x = FieldRegister::<E::BaseField>::from_register(
            self.get_local_memory(E::BaseField::NB_LIMBS),
        );
        let y = FieldRegister::<E::BaseField>::from_register(
            self.get_local_memory(E::BaseField::NB_LIMBS),
        );
        let z = FieldRegister::<E::BaseField>::from_register(
            self.get_local_memory(E::BaseField::NB_LIMBS),
        );
        SWProjectivePointRegister::new(x, y, z)
    }

    fn alloc_local_sw_point(&mut self) -> SWProjectivePointRegister<E> {
        let x = self.alloc::<FieldRegister<E::BaseField>>();
        let y = self.alloc::<FieldRegister<E::BaseField>>();
        let z = self.alloc::<FieldRegister<E::BaseField>>();
        SWProjectivePointRegister::new(x, y, z)
    }

    fn alloc_next_sw_point(&mut self) -> SWProjectivePointRegister<E> {
        let x = self.alloc::<FieldRegister<E::BaseField>>();
        let y = self.alloc::<FieldRegister<E::BaseField>>();
        let z = self.alloc::<FieldRegister<E::BaseField>>();
        SWProjectivePointRegister::new(x, y, z)
    }

    fn alloc_global_sw_point(&mut self) -> SWProjectivePointRegister<E> {
        let x = self.alloc::<FieldRegister<E::BaseField>>();
        let y = self.alloc::<FieldRegister<E::BaseField>>();
        let z = self.alloc::<FieldRegister<E::BaseField>>();
        SWProjectivePointRegister::new(x, y, z)
    }

    fn alloc_public_sw_point(&mut self) -> SWProjectivePointRegister<E> {
        let x = self.alloc::<FieldRegister<E::BaseField>>();
        let y = self.alloc::<FieldRegister<E::BaseField>>();
        let z = self.alloc::<FieldRegister<E::BaseField>>();
        SWProjectivePointRegister::new(x, y, z)
    }
}

impl<F: PrimeField64, E: EllipticCurveParameters> SWEllipticCurveWriter<E> for TraceWriter<F> {
    fn read_sw_point(
        &self,
        data: &SWProjectivePointRegister<E>,
        row_index: usize,
    ) -> SWProjectivePoint<E> {
        let p_x = self.read(&data.x, row_index);
        let p_y = self.read(&data.y, row_index);
        let p_z = self.read(&data.z, row_index);

        let x = field_limbs_to_biguint(p_x.coefficients());
        let y = field_limbs_to_biguint(p_y.coefficients());
        let z = field_limbs_to_biguint(p_z.coefficients());

        SWProjectivePoint::<E>::new(x, y, z)
    }

    fn write_sw_point(
        &self,
        data: &SWProjectivePointRegister<E>,
        value: &SWProjectivePoint<E>,
        row_index: usize,
    ) {
        let value_x = to_u16_le_limbs_polynomial::<F, E::BaseField>(&value.x);
        let value_y = to_u16_le_limbs_polynomial::<F, E::BaseField>(&value.y);
        let value_z = to_u16_le_limbs_polynomial::<F, E::BaseField>(&value.z);
        self.write(&data.x, &value_x, row_index);
        self.write(&data.y, &value_y, row_index);
        self.write(&data.z, &value_z, row_index);
    }
}
