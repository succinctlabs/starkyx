use super::element::CubicElement;
use crate::curta::air::parser::AirParser;

#[derive(Debug)]
pub struct CubicParser<'a, AP: AirParser> {
    pub parser: &'a mut AP,
}

impl<'a, AP: AirParser> CubicParser<'a, AP> {
    pub fn new(parser: &'a mut AP) -> Self {
        Self { parser }
    }

    pub fn constant(&mut self, a: [AP::Field; 3]) -> CubicElement<AP::Var> {
        CubicElement([
            self.parser.constant(a[0]),
            self.parser.constant(a[1]),
            self.parser.constant(a[2]),
        ])
    }

    pub fn zero(&mut self) -> CubicElement<AP::Var> {
        CubicElement([self.parser.zero(), self.parser.zero(), self.parser.zero()])
    }

    pub fn one_extension(&mut self) -> CubicElement<AP::Var> {
        CubicElement([self.parser.one(), self.parser.zero(), self.parser.zero()])
    }

    pub fn from_base(&mut self, element: AP::Var) -> CubicElement<AP::Var> {
        CubicElement([element, self.parser.zero(), self.parser.zero()])
    }

    pub fn add(
        &mut self,
        a: &CubicElement<AP::Var>,
        b: &CubicElement<AP::Var>,
    ) -> CubicElement<AP::Var> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([
            self.parser.add(x_0, y_0),
            self.parser.add(x_1, y_1),
            self.parser.add(x_2, y_2),
        ])
    }

    pub fn sub(
        &mut self,
        a: &CubicElement<AP::Var>,
        b: &CubicElement<AP::Var>,
    ) -> CubicElement<AP::Var> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);
        CubicElement([
            self.parser.sub(x_0, y_0),
            self.parser.sub(x_1, y_1),
            self.parser.sub(x_2, y_2),
        ])
    }

    pub fn mul(
        &mut self,
        a: &CubicElement<AP::Var>,
        b: &CubicElement<AP::Var>,
    ) -> CubicElement<AP::Var> {
        let (x_0, x_1, x_2) = (a.0[0], a.0[1], a.0[2]);
        let (y_0, y_1, y_2) = (b.0[0], b.0[1], b.0[2]);

        let x_0y_0 = self.parser.mul(x_0, y_0);
        let x_0y_1 = self.parser.mul(x_0, y_1);
        let x_0y_2 = self.parser.mul(x_0, y_2);
        let x_1y_0 = self.parser.mul(x_1, y_0);
        let x_1y_1 = self.parser.mul(x_1, y_1);
        let x_1y_2 = self.parser.mul(x_1, y_2);
        let x_2y_0 = self.parser.mul(x_2, y_0);
        let x_2y_1 = self.parser.mul(x_2, y_1);
        let x_2y_2 = self.parser.mul(x_2, y_2);

        let mut z_0 = self.parser.sub(x_0y_0, x_1y_2);
        z_0 = self.parser.sub(z_0, x_2y_1);

        let mut z_1 = self.parser.add(x_0y_1, x_1y_0);
        z_1 = self.parser.add(z_1, x_1y_2);
        z_1 = self.parser.add(z_1, x_2y_1);
        z_1 = self.parser.sub(z_1, x_2y_2);

        let mut z_2 = self.parser.add(x_0y_2, x_1y_1);
        z_2 = self.parser.add(z_2, x_2y_0);
        z_2 = self.parser.add(z_2, x_2y_2);

        CubicElement([z_0, z_1, z_2])
    }
}
