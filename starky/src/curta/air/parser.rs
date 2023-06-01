use plonky2::field::types::Field;

pub trait AirParser: Send + Sized {
    type Field: Field;

    type Var: Copy;

    fn constant(&mut self, value: Self::Field) -> Self::Var;

    fn local_slice(&self) -> &[Self::Var];
    fn next_slice(&self) -> &[Self::Var];
    fn challenge_slice(&self) -> &[Self::Var];
    fn public_slice(&self) -> &[Self::Var];

    fn constraint(&mut self, constraint: Self::Var);
    fn constraint_transition(&mut self, constraint: Self::Var);
    fn constraint_first_row(&mut self, constraint: Self::Var);
    fn constraint_last_row(&mut self, constraint: Self::Var);

    /// Add two vars while potantially updating the internal state
    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    fn neg(&mut self, a: Self::Var) -> Self::Var;

    /// Multiply two vars while potantially updating the internal state
    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    fn scalar_add(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.add(a, b)
    }

    fn scalar_sub(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.sub(a, b)
    }

    fn scalar_mul(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.mul(a, b)
    }

    fn one(&mut self) -> Self::Var {
        self.constant(Self::Field::ONE)
    }

    fn zero(&mut self) -> Self::Var {
        self.constant(Self::Field::ZERO)
    }

    fn assert_eq(&mut self, a: Self::Var, b: Self::Var) {
        let c = self.sub(a, b);
        self.constraint(c);
    }

    fn assert_eq_transition(&mut self, a: Self::Var, b: Self::Var) {
        let c = self.sub(a, b);
        self.constraint_transition(c);
    }
}
