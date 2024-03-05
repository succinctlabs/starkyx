use serde::{Deserialize, Serialize};

use crate::air::extension::cubic::CubicParser;
use crate::chip::builder::AirBuilder;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::cubic::CubicRegister;
use crate::chip::register::Register;
use crate::chip::trace::writer::TraceWriter;
use crate::math::prelude::*;
use crate::prelude::cubic::element::CubicElement;
use crate::prelude::{AirConstraint, AirParameters};

/// Powers of a challenge element.
///
/// This struct defines the constraints and writing method to get the powers of a challenge element
/// meant for Reed-Solomon fingerprinting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Powers<F, E> {
    element: CubicRegister,
    values: ArrayRegister<CubicRegister>,
    _marker: std::marker::PhantomData<(F, E)>,
}

impl<L: AirParameters> AirBuilder<L> {
    /// Get an array of powers 1, `gamma`,..., `gamma^{len-1}` of a verifier challenge.
    pub fn challenge_powers(&mut self, len: usize) -> ArrayRegister<CubicRegister> {
        let challenge = self.alloc_challenge();
        let power_values = self.alloc_array_global(len);

        let powers = Powers {
            element: challenge,
            values: power_values,
            _marker: std::marker::PhantomData,
        };

        self.powers.push(powers.clone());
        self.global_constraints.push(powers.into());

        power_values
    }
}

impl<E: CubicParameters<AP::Field>, AP: CubicParser<E>> AirConstraint<AP> for Powers<AP::Field, E> {
    fn eval(&self, parser: &mut AP) {
        let element = self.element.eval(parser);
        let powers = self.values.eval_vec(parser);

        if powers.is_empty() {
            return;
        }

        let one = parser.one_extension();
        parser.assert_eq_extension(powers[0], one);

        for window in powers.windows(2) {
            let left = window[0];
            let right = window[1];
            let left_times_element = parser.mul_extension(left, element);
            parser.assert_eq_extension(left_times_element, right);
        }
    }
}

impl<F: Field> TraceWriter<F> {
    pub fn write_powers<E: CubicParameters<F>>(&self, powers: &Powers<F, E>) {
        let elememt = self.read(&powers.element, 0);

        let mut power = CubicElement::ONE;
        for power_reg in powers.values.iter() {
            self.write(&power_reg, &power, 0);
            power *= elememt;
        }
    }
}
