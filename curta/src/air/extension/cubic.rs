use crate::air::parser::AirParser;
use crate::math::extension::cubic::element::CubicElement;
use crate::math::extension::cubic::extension::CubicExtension;
use crate::math::extension::cubic::parameters::CubicParameters;

pub trait CubicParser<E: CubicParameters<Self::Field>>: AirParser {
    fn element_from_base_field(&mut self, value: Self::Var) -> CubicElement<Self::Var> {
        CubicElement([value, self.zero(), self.zero()])
    }

    fn element_from_base_slice(&self, values: &[Self::Var]) -> CubicElement<Self::Var> {
        assert!(values.len() == 3);
        CubicElement([values[0], values[1], values[2]])
    }

    fn as_base_array(&self, value: CubicElement<Self::Var>) -> [Self::Var; 3] {
        value.0
    }

    fn one_extension(&mut self) -> CubicElement<Self::Var> {
        CubicElement([self.one(), self.zero(), self.zero()])
    }

    fn zero_extension(&mut self) -> CubicElement<Self::Var> {
        CubicElement([self.zero(), self.zero(), self.zero()])
    }

    fn constant_extension(
        &mut self,
        value: CubicExtension<Self::Field, E>,
    ) -> CubicElement<Self::Var> {
        let CubicElement([x_0, x_1, x_2]) = value.0;
        CubicElement([self.constant(x_0), self.constant(x_1), self.constant(x_2)])
    }

    fn add_extension(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) -> CubicElement<Self::Var> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([self.add(x_0, y_0), self.add(x_1, y_1), self.add(x_2, y_2)])
    }

    fn add_many_extension(
        &mut self,
        elements: &[CubicElement<Self::Var>],
    ) -> CubicElement<Self::Var> {
        let mut sum = self.zero_extension();
        for element in elements {
            sum = self.add_extension(sum, *element);
        }
        sum
    }

    fn sub_extension(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) -> CubicElement<Self::Var> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([self.sub(x_0, y_0), self.sub(x_1, y_1), self.sub(x_2, y_2)])
    }

    fn mul_extension(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) -> CubicElement<Self::Var> {
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

    fn scalar_mul_extension(
        &mut self,
        a: CubicElement<Self::Var>,
        scalar: Self::Var,
    ) -> CubicElement<Self::Var> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        CubicElement([
            self.mul(x_0, scalar),
            self.mul(x_1, scalar),
            self.mul(x_2, scalar),
        ])
    }

    fn neg_extension(&mut self, a: CubicElement<Self::Var>) -> CubicElement<Self::Var> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        CubicElement([self.neg(x_0), self.neg(x_1), self.neg(x_2)])
    }

    fn constraint_extension(&mut self, a: CubicElement<Self::Var>) {
        let a_arr = self.as_base_array(a);
        for a in a_arr {
            self.constraint(a);
        }
    }

    fn constraint_extension_transition(&mut self, a: CubicElement<Self::Var>) {
        let a_arr = self.as_base_array(a);
        for a in a_arr {
            self.constraint_transition(a);
        }
    }

    fn constraint_extension_first_row(&mut self, a: CubicElement<Self::Var>) {
        let a_arr = self.as_base_array(a);
        for a in a_arr {
            self.constraint_first_row(a);
        }
    }

    fn constraint_extension_last_row(&mut self, a: CubicElement<Self::Var>) {
        let a_arr = self.as_base_array(a);
        for a in a_arr {
            self.constraint_last_row(a);
        }
    }

    fn assert_eq_extension(&mut self, a: CubicElement<Self::Var>, b: CubicElement<Self::Var>) {
        let c = self.sub_extension(a, b);
        self.constraint_extension(c);
    }

    fn assert_eq_extension_first_row(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) {
        let c = self.sub_extension(a, b);
        self.constraint_extension_first_row(c);
    }

    fn assert_eq_extension_last_row(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) {
        let c = self.sub_extension(a, b);
        self.constraint_extension_last_row(c);
    }

    fn assert_eq_extension_transition(
        &mut self,
        a: CubicElement<Self::Var>,
        b: CubicElement<Self::Var>,
    ) {
        let c = self.sub_extension(a, b);
        self.constraint_extension_transition(c);
    }
}
