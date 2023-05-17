use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy)]
pub struct CubicArray<T>(pub [T; 3]);

impl<T: Copy> CubicArray<T> {
    pub fn new(a: T, b: T, c: T) -> Self {
        Self([a, b, c])
    }

    pub fn from_base(element: T, zero: T) -> Self {
        Self([element, zero, zero])
    }

    pub fn from_slice(slice: &[T]) -> Self {
        assert_eq!(slice.len(), 3, "Cubic array slice must have length 3");
        Self([slice[0], slice[1], slice[2]])
    }
}

impl<T: Copy + Add<Output = T>> Add for CubicArray<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl<T: Copy + Sub<Output = T>> Sub for CubicArray<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl<T: Copy + Neg<Output = T>> Neg for CubicArray<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl<T: Copy + AddAssign> AddAssign for CubicArray<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
    }
}

impl<T: Copy + SubAssign> SubAssign for CubicArray<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T>> Mul for CubicArray<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (x_0, x_1, x_2) = (self.0[0], self.0[1], self.0[2]);
        let (y_0, y_1, y_2) = (rhs.0[0], rhs.0[1], rhs.0[2]);

        // Using u^3 = u-1 we get:
        // (x_0 + x_1 u + x_2 u^2) * (y_0 + y_1 u + y_2 u^2)
        // = (x_0y_0 - x_1y_2 - x_2y_1)
        // + (x_0y_1 + x_1y_0 + x_1y_2 + x_2y_1) u
        // + (x_0y_2 + x_1y_1 + x_2y_0) u^2
        Self([
            x_0 * y_0 - x_1 * y_2 - x_2 * y_1,
            x_0 * y_1 + x_1 * y_0 + x_1 * y_2 + x_2 * y_1 - x_2 * y_2,
            x_0 * y_2 + x_1 * y_1 + x_2 * y_0 + x_2 * y_2,
        ])
    }
}
