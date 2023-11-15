use core::fmt::Debug;

use super::extension::cubic::CubicParser;
use crate::math::prelude::*;

/// A general interface for interacting with AIR constraints.
///
/// This trait is used to abstract over the different types of ways to compile and evaluate AIR
/// constraints. This trait is used to create a unified interface for writing constraints which can
/// then be consumed by the prover, the verifier, and a recursive constraint compiler, all of which
/// would use a different implementation of this trait.
///
/// ### Background
/// AIR constraint systems were introduced by [Ben-Sasson et al.](https://eprint.iacr.org/2018/046.pdf)
/// as part of the STARK protocol. In this protocol, the prover seeks to convince the verifier that
/// a given table of values consitutes a valid computation trace of a given program. The semantics
/// of the program are encoded as polynomial equations via the AIR constraints.  
///
/// An AIR table is a matrix of field elements of dimenstions `n x k` with `n` being the number of
/// rows and `k` the number of columns. For a table instance `X` we will mark the element in
/// location `(i, j)` by `X[i,j]`. AIR constraints are polynomial constraints consiting of three
/// multivariable polynomials `P_first_row`, `P_transition`, and `P_last_row` that specify first row
///  constraints, transition constraints, and last row constraints. Exlicitly, a table instance `X`
/// satisfies the AIR constraints if and only if the following three conditions hold:
///    - *First row*:`P_first_row[X[0, 0], ..., X[0, k-1]] = 0`.
///    - *Last row*: `P_last_row[X[n-1, 0], ..., X[n-1, k-1]] = 0`.
///    - *Transtion*: `P_transition[X[i, 0], ..., X[i, i, k-1], X[i+1, 0], .. , X[i+1, k-1]] = 0`
///                   for all `i` in `[0, n-2]`.
/// One can think of the AIR constraints as specifying a state machine, with the first row and last
///  row constraints as boundary conditions and the transition constraints as the state transition
/// function. For more information about AIR and STARKs in general see:
///   - [STARK: Scalable Transparent Arguments of Knowledge](https://eprint.iacr.org/2018/046.pdf)
///   - [ethSTARK Documentation](https://eprint.iacr.org/2021/582.pdf)
///   - [Anatomy of a STARK](https://aszepieniec.github.io/stark-anatomy/)
///
/// ### Random `AIR` with challenges
/// The `AirParser` trait provides a general interface for encoding AIR constraints that include
/// random challenges. For this reason, there are two slices of public inputs: the public slice w
/// which consists of public inputs for the constraint system, and a global slice which consists of
/// values produced by the prover after it recevied challenges from the verifier.
///
/// ### Encoding constraints with an `AirParser`
/// The `AirParser` trait provides a general interface for encoding AIR constraints. The trait
/// consists of three types of functions:
///  - *data access*: functions that return slices of either the local row, the next row, the
///    public slice row, the challenge row, or the global row.
///  - *constraint*: functions that register a constraint into the constraint system, which could
///    be either a first row constraint, transition constraint, or a last row constraint.
/// - *arithmetic*: functions that perform arithmetic operations on variables.
pub trait AirParser: Sized {
    /// The underlying base field of the constraint system.
    type Field: Field;
    /// The underlying variable type of the constraint system.
    type Var: Debug + Copy + 'static;

    /// Returns a slice representing elements of the current row of the AIR table.
    fn local_slice(&self) -> &[Self::Var];
    /// Returns a slice representing elements of the next row of the AIR table.
    fn next_slice(&self) -> &[Self::Var];
    /// Returns a slice representing challenges from the verifier.
    fn challenge_slice(&self) -> &[Self::Var];
    /// Returns a slice representing public inputs to the constraint system.
    fn public_slice(&self) -> &[Self::Var];
    /// Returns a slice representing values written by the prover after receiving challenges.
    fn global_slice(&self) -> &[Self::Var];

    /// Assert that `constraint` is zero.
    fn constraint(&mut self, constraint: Self::Var);
    /// Assert that `constraint` is zero in the transition constraints.
    ///
    /// This means that the constraint is only asserted if the current row is not the last row.
    fn constraint_transition(&mut self, constraint: Self::Var);
    /// Assert that `constraint` is zero in the first row constraints.
    fn constraint_first_row(&mut self, constraint: Self::Var);
    /// Assert that `constraint` is zero in the last row constraints.
    fn constraint_last_row(&mut self, constraint: Self::Var);

    /// Create a variable representing a constant value.
    fn constant(&mut self, value: Self::Field) -> Self::Var;

    /// Add two vars while potantially updating the internal state
    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    /// Subtract two vars while potantially updating the internal state
    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    /// Negate a var while potantially updating the internal state
    fn neg(&mut self, a: Self::Var) -> Self::Var;

    /// Multiply two vars while potantially updating the internal state
    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var;

    /// Add a constant to a var while potantially updating the internal state
    fn add_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.add(a, b)
    }

    fn sub_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.sub(a, b)
    }

    fn mul_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        let b = self.constant(b);
        self.mul(a, b)
    }

    fn one(&mut self) -> Self::Var {
        self.constant(Self::Field::ONE)
    }

    fn zero(&mut self) -> Self::Var {
        self.constant(Self::Field::ZERO)
    }

    fn sum(&mut self, elements: &[Self::Var]) -> Self::Var {
        elements
            .iter()
            .fold(self.zero(), |acc, x| self.add(acc, *x))
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

#[derive(Debug)]
pub struct MulParser<'a, AP: AirParser> {
    pub parser: &'a mut AP,
    pub multiplier: AP::Var,
}

impl<'a, AP: AirParser> MulParser<'a, AP> {
    pub fn new(parser: &'a mut AP, multiplier: AP::Var) -> Self {
        Self { parser, multiplier }
    }
}

impl<'a, AP: AirParser> AirParser for MulParser<'a, AP> {
    type Field = AP::Field;
    type Var = AP::Var;

    fn local_slice(&self) -> &[Self::Var] {
        self.parser.local_slice()
    }

    fn next_slice(&self) -> &[Self::Var] {
        self.parser.next_slice()
    }

    fn challenge_slice(&self) -> &[Self::Var] {
        self.parser.challenge_slice()
    }

    fn global_slice(&self) -> &[Self::Var] {
        self.parser.global_slice()
    }

    fn public_slice(&self) -> &[Self::Var] {
        self.parser.public_slice()
    }

    fn constraint(&mut self, constraint: Self::Var) {
        let constr = self.parser.mul(constraint, self.multiplier);
        self.parser.constraint(constr);
    }

    fn constraint_transition(&mut self, constraint: Self::Var) {
        let constr = self.parser.mul(constraint, self.multiplier);
        self.parser.constraint_transition(constr);
    }

    fn constraint_first_row(&mut self, constraint: Self::Var) {
        let constr = self.parser.mul(constraint, self.multiplier);
        self.parser.constraint_first_row(constr);
    }

    fn constraint_last_row(&mut self, constraint: Self::Var) {
        let constr = self.parser.mul(constraint, self.multiplier);
        self.parser.constraint_last_row(constr);
    }

    fn constant(&mut self, value: Self::Field) -> Self::Var {
        self.parser.constant(value)
    }

    fn add(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.parser.add(a, b)
    }

    fn sub(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.parser.sub(a, b)
    }

    fn neg(&mut self, a: Self::Var) -> Self::Var {
        self.parser.neg(a)
    }

    fn mul(&mut self, a: Self::Var, b: Self::Var) -> Self::Var {
        self.parser.mul(a, b)
    }

    fn add_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        self.parser.add_const(a, b)
    }

    fn sub_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        self.parser.sub_const(a, b)
    }

    fn mul_const(&mut self, a: Self::Var, b: Self::Field) -> Self::Var {
        self.parser.mul_const(a, b)
    }
}

// TODO: implement parser specific functions
impl<'a, AP: CubicParser<E>, E: CubicParameters<AP::Field>> CubicParser<E> for MulParser<'a, AP> {}
