use super::AirBuilder;
use crate::chip::constraint::arithmetic::expression::ArithmeticExpression;
use crate::chip::constraint::arithmetic::ArithmeticConstraint;
use crate::chip::instruction::assign::{AssignInstruction, AssignType};
use crate::chip::instruction::set::InstructionSet;
use crate::chip::register::Register;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    #[inline]
    pub fn assert_expression_zero(&mut self, expression: ArithmeticExpression<L::Field>) {
        let constraint = ArithmeticConstraint::All(expression);
        self.constraints.push(constraint.into());
    }

    #[inline]
    pub fn assert_expression_zero_first_row(&mut self, expression: ArithmeticExpression<L::Field>) {
        let constraint = ArithmeticConstraint::First(expression);
        self.constraints.push(constraint.into());
    }

    #[inline]
    pub fn assert_expression_zero_last_row(&mut self, expression: ArithmeticExpression<L::Field>) {
        let constraint = ArithmeticConstraint::Last(expression);
        self.constraints.push(constraint.into());
    }

    #[inline]
    pub fn assert_expression_zero_transition(
        &mut self,
        expression: ArithmeticExpression<L::Field>,
    ) {
        let constraint = ArithmeticConstraint::Transition(expression);
        self.constraints.push(constraint.into());
    }

    #[inline]
    pub fn assert_expressions_equal(
        &mut self,
        a: ArithmeticExpression<L::Field>,
        b: ArithmeticExpression<L::Field>,
    ) {
        let constraint = ArithmeticConstraint::All(a - b);
        self.constraints.push(constraint.into());
    }

    #[inline]
    pub fn assert_expressions_equal_first_row(
        &mut self,
        a: ArithmeticExpression<L::Field>,
        b: ArithmeticExpression<L::Field>,
    ) {
        let constraint = ArithmeticConstraint::First(a - b);
        self.constraints.push(constraint.into());
    }

    #[inline]
    pub fn assert_expressions_equal_last_row(
        &mut self,
        a: ArithmeticExpression<L::Field>,
        b: ArithmeticExpression<L::Field>,
    ) {
        let constraint = ArithmeticConstraint::Last(a - b);
        self.constraints.push(constraint.into());
    }

    #[inline]
    pub fn assert_expressions_equal_transition(
        &mut self,
        a: ArithmeticExpression<L::Field>,
        b: ArithmeticExpression<L::Field>,
    ) {
        let constraint = ArithmeticConstraint::Transition(a - b);
        self.constraints.push(constraint.into());
    }

    #[inline]
    pub fn assert_equal<T: Register>(&mut self, a: &T, b: &T) {
        self.assert_expression_zero(a.expr() - b.expr());
    }

    #[inline]
    pub fn assert_equal_first_row<T: Register>(&mut self, a: &T, b: &T) {
        self.assert_expression_zero_first_row(a.expr() - b.expr());
    }

    #[inline]
    pub fn assert_equal_last_row<T: Register>(&mut self, a: &T, b: &T) {
        self.assert_expression_zero_last_row(a.expr() - b.expr());
    }

    #[inline]
    pub fn assert_equal_transition<T: Register>(&mut self, a: &T, b: &T) {
        self.assert_expression_zero_transition(a.expr() - b.expr());
    }

    #[inline]
    pub fn set_to_expression<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) {
        let instr = AssignInstruction::new(expression.clone(), *data.register(), AssignType::All);
        self.register_from_instruction_set(InstructionSet::Assign(instr));
    }

    #[inline]
    pub fn set_to_expression_first_row<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) {
        let instr = AssignInstruction::new(expression.clone(), *data.register(), AssignType::First);
        self.register_from_instruction_set(InstructionSet::Assign(instr));
    }

    #[inline]
    pub fn set_to_expression_last_row<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) {
        let instr = AssignInstruction::new(expression.clone(), *data.register(), AssignType::Last);
        self.register_from_instruction_set(InstructionSet::Assign(instr));
    }

    #[inline]
    pub fn set_to_expression_transition<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) {
        let instr =
            AssignInstruction::new(expression.clone(), *data.register(), AssignType::Transition);
        self.register_from_instruction_set(InstructionSet::Assign(instr));
    }
}
