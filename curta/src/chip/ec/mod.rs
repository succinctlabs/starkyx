use core::fmt::Debug;

use self::point::AffinePointRegister;
use super::builder::AirBuilder;
use super::field::parameters::FieldParameters;
use super::AirParameters;

pub mod edwards;
pub mod gadget;
pub mod point;
pub mod weierstrass;

pub trait EllipticCurveParameters: Debug + Send + Sync + Copy + 'static {
    type BaseField: FieldParameters;
}

pub trait EllipticCurve<L: AirParameters>: EllipticCurveParameters {
    fn ec_add(
        builder: &mut AirBuilder<L>,
        p: &AffinePointRegister<Self>,
        q: &AffinePointRegister<Self>,
    ) -> AffinePointRegister<Self>;

    fn ec_double(
        builder: &mut AirBuilder<L>,
        p: &AffinePointRegister<Self>,
    ) -> AffinePointRegister<Self>;
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn ec_add<E: EllipticCurve<L>>(
        &mut self,
        p: &AffinePointRegister<E>,
        q: &AffinePointRegister<E>,
    ) -> AffinePointRegister<E> {
        E::ec_add(self, p, q)
    }

    pub fn ec_double<E: EllipticCurve<L>>(
        &mut self,
        p: &AffinePointRegister<E>,
    ) -> AffinePointRegister<E> {
        E::ec_double(self, p)
    }
}
