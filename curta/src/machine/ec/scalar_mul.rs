use crate::chip::ec::point::AffinePointRegister;
use crate::chip::ec::scalar::ECScalarRegister;
use crate::chip::ec::EllipticCurve;

pub struct ScalarMulData<E: EllipticCurve> {
    points: Vec<AffinePointRegister<E>>,
    scalars: Vec<ECScalarRegister<E>>,
}
