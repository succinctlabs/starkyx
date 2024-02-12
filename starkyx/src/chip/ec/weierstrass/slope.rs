use super::{SWCurve, WeierstrassParameters};
use crate::chip::builder::AirBuilder;
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::register::FieldRegister;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    /// Given two points `p` and `q`, compute the slope of the line passing through them.
    ///
    /// The slope is given by the formula `(q.y - p.y) / (q.x - p.x)`. This function assumes that
    /// `p` and `q` are different points
    pub(crate) fn sw_slope_different<E: WeierstrassParameters>(
        &mut self,
        p: &AffinePointRegister<SWCurve<E>>,
        q: &AffinePointRegister<SWCurve<E>>,
    ) -> FieldRegister<E::BaseField>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        let (x_1, y_1) = (p.x, p.y);
        let (x_2, y_2) = (q.x, q.y);

        let slope_numerator = self.fp_sub(&y_2, &y_1);
        let slope_denominator = self.fp_sub(&x_2, &x_1);

        self.fp_div(&slope_numerator, &slope_denominator)
    }

    /// Given a point `p`, compute the slope of the tangent line at `p`.
    ///
    /// The slope is given by the formula `(3 * p.x^2 + a) / (2 * p.y)`.
    pub(crate) fn sw_tangent<E: WeierstrassParameters>(
        &mut self,
        p: &AffinePointRegister<SWCurve<E>>,
        a: &FieldRegister<E::BaseField>,
        three: &FieldRegister<E::BaseField>,
    ) -> FieldRegister<E::BaseField>
    where
        L::Instruction: FromFieldInstruction<E::BaseField>,
    {
        let (x_1, y_1) = (p.x, p.y);

        let x_1_sq = self.fp_mul(&x_1, &x_1);
        let x_1_sq_3 = self.fp_mul(&x_1_sq, three);
        let slope_numerator = self.fp_add(&x_1_sq_3, a);
        let slope_denominator = self.fp_add(&y_1, &y_1);

        self.fp_div(&slope_numerator, &slope_denominator)
    }
}
