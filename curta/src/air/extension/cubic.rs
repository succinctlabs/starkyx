use super::ExtensionParser;
use crate::air::parser::AirParser;
use crate::math::extension::cubic::parameters::CubicParameters;
use crate::plonky2::field::cubic::element::CubicElement;
use crate::plonky2::field::cubic::extension::CubicExtension;

pub trait CubicParser<E: CubicParameters<Self::Field>>: AirParser {}

impl<AP: CubicParser<E>, E: CubicParameters<AP::Field>>
    ExtensionParser<CubicExtension<AP::Field, E>> for AP
{
    type ExtensionVar = CubicElement<AP::Var>;

    fn from_base_field(&mut self, value: Self::Var) -> Self::ExtensionVar {
        CubicElement([value, self.zero(), self.zero()])
    }

    fn one_extension(&mut self) -> Self::ExtensionVar {
        CubicElement([self.one(), self.zero(), self.zero()])
    }

    fn zero_extension(&mut self) -> Self::ExtensionVar {
        CubicElement([self.zero(), self.zero(), self.zero()])
    }

    fn constant_extension(&mut self, value: CubicExtension<AP::Field, E>) -> Self::ExtensionVar {
        let CubicElement([x_0, x_1, x_2]) = value.0;
        CubicElement([self.constant(x_0), self.constant(x_1), self.constant(x_2)])
    }

    fn add_extension(
        &mut self,
        a: Self::ExtensionVar,
        b: Self::ExtensionVar,
    ) -> Self::ExtensionVar {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([self.add(x_0, y_0), self.add(x_1, y_1), self.add(x_2, y_2)])
    }

    fn sub_extension(
        &mut self,
        a: Self::ExtensionVar,
        b: Self::ExtensionVar,
    ) -> Self::ExtensionVar {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([self.sub(x_0, y_0), self.sub(x_1, y_1), self.sub(x_2, y_2)])
    }

    fn mul_extension(
        &mut self,
        a: Self::ExtensionVar,
        b: Self::ExtensionVar,
    ) -> Self::ExtensionVar {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);

        let x_0y_0 = self.mul(x_0, y_0);
        let x_0y_1 = self.mul(x_0, y_1);
        let x_0y_2 = self.mul(x_0, y_2);
        let x_1y_0 = self.mul(x_1, y_0);
        let x_1y_1 = self.mul(x_1, y_1);
        let x_1y_2 = self.mul(x_1, y_2);
        let x_2y_0 = self.mul(x_2, y_0);
        let x_2y_1 = self.mul(x_2, y_1);
        let x_2y_2 = self.mul(x_2, y_2);

        let mut z_0 = self.sub(x_0y_0, x_1y_2);
        z_0 = self.sub(z_0, x_2y_1);

        let mut z_1 = self.add(x_0y_1, x_1y_0);
        z_1 = self.add(z_1, x_1y_2);
        z_1 = self.add(z_1, x_2y_1);
        z_1 = self.sub(z_1, x_2y_2);

        let mut z_2 = self.add(x_0y_2, x_1y_1);
        z_2 = self.add(z_2, x_2y_0);
        z_2 = self.add(z_2, x_2y_2);

        CubicElement([z_0, z_1, z_2])
    }

    fn neg_extension(&mut self, a: Self::ExtensionVar) -> Self::ExtensionVar {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        CubicElement([self.neg(x_0), self.neg(x_1), self.neg(x_2)])
    }
}
