use crate::chip::ec::point::AffinePointRegister;
use crate::chip::ec::scalar::ECScalarRegister;
use crate::chip::ec::EllipticCurve;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::machine::builder::Builder;

pub trait EllipticCurveBuilder<E: EllipticCurve> {
    fn alloc_ec_point(&mut self) -> AffinePointRegister<E>;
    fn alloc_public_ec_point(&mut self) -> AffinePointRegister<E>;
    // fn load_bits(&mut self, scalars: &[ECScalarRegister<E>]) -> BitRegister;
}

impl<E: EllipticCurve, B: Builder> EllipticCurveBuilder<E> for B {
    fn alloc_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = self.alloc();
        let y = self.alloc();

        AffinePointRegister::new(x, y)
    }

    fn alloc_public_ec_point(&mut self) -> AffinePointRegister<E> {
        let x = self.alloc_public();
        let y = self.alloc_public();

        AffinePointRegister::new(x, y)
    }

    // fn load_bits(&mut self, scalars: &[ECScalarRegister<E>]) -> BitRegister {
    //     let limb_slice = self.uninit_slice::<ElementRegister>();

    //     for (i, limb) in scalars.iter().flat_map(|s| s.limbs().iter()).enumerate() {}

    //     todo!()
    // }
}
