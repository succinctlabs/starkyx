use serde::{Deserialize, Serialize};

use self::expression::ArithmeticExpression;
use crate::air::parser::AirParser;
use crate::air::AirConstraint;

pub mod expression;
pub(crate) mod expression_slice;

use crate::math::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ArithmeticConstraint<F> {
    First(ArithmeticExpression<F>),
    Last(ArithmeticExpression<F>),
    Transition(ArithmeticExpression<F>),
    All(ArithmeticExpression<F>),
}

impl<F: Field, AP: AirParser<Field = F>> AirConstraint<AP> for ArithmeticConstraint<F> {
    fn eval(&self, parser: &mut AP) {
        match self {
            ArithmeticConstraint::First(expression) => {
                let constraints = expression.eval(parser);
                for constraint in constraints {
                    parser.constraint_first_row(constraint);
                }
            }
            ArithmeticConstraint::Last(expression) => {
                let constraints = expression.eval(parser);
                for constraint in constraints {
                    parser.constraint_last_row(constraint);
                }
            }
            ArithmeticConstraint::Transition(expression) => {
                let constraints = expression.eval(parser);
                for constraint in constraints {
                    parser.constraint_transition(constraint);
                }
            }
            ArithmeticConstraint::All(expression) => {
                let constraints = expression.eval(parser);
                for constraint in constraints {
                    parser.constraint(constraint);
                }
            }
        }
    }
}
