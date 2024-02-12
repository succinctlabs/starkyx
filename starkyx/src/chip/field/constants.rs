use num::{BigUint, One, Zero};

use super::parameters::FieldParameters;
use super::register::FieldRegister;
use crate::chip::builder::AirBuilder;
use crate::chip::AirParameters;
use crate::polynomial::Polynomial;

impl<L: AirParameters> AirBuilder<L> {
    pub fn fp_constant<P: FieldParameters>(&mut self, num: &BigUint) -> FieldRegister<P> {
        let poly =
            Polynomial::<L::Field>::from_biguint_field(num, P::NB_BITS_PER_LIMB, P::NB_LIMBS);

        self.constant(&poly)
    }

    pub fn fp_zero<P: FieldParameters>(&mut self) -> FieldRegister<P> {
        self.fp_constant(&BigUint::zero())
    }

    pub fn fp_one<P: FieldParameters>(&mut self) -> FieldRegister<P> {
        self.fp_constant(&BigUint::one())
    }
}
