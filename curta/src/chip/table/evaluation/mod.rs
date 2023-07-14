use core::marker::PhantomData;

use crate::chip::builder::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::constraint::Constraint;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::extension::ExtensionRegister;
use crate::chip::AirParameters;
use crate::math::prelude::*;

pub mod constraint;
pub mod trace;

#[derive(Debug, Clone)]
pub struct Evaluation<F: Field, E: CubicParameters<F>> {
    beta: ExtensionRegister<3>,
    beta_powers: ExtensionRegister<3>,
    alphas: ArrayRegister<ExtensionRegister<3>>,
    values: Vec<ElementRegister>,
    filter: ArithmeticExpression<F>,
    _marker: PhantomData<(F, E)>,
}

impl<L: AirParameters> AirBuilder<L> {
    pub fn evaluation(
        &mut self,
        values: &[ElementRegister],
        filter: ArithmeticExpression<L::Field>,
    ) {
        // Get the running evaluation challenge
        let beta = self.alloc_challenge::<ExtensionRegister<3>>();
        let beta_powers = self.alloc::<ExtensionRegister<3>>();
        // get the row accumulation challenge
        let alphas = self.alloc_challenge_array::<ExtensionRegister<3>>(values.len());

        let evaluation = Evaluation {
            beta,
            beta_powers,
            alphas,
            values: values.to_vec(),
            filter,
            _marker: PhantomData,
        };
        self.constraints
            .push(Constraint::evaluation(evaluation.clone()));
        self.evaluation_data.push(evaluation);
    }
}
