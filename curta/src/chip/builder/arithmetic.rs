use super::AirBuilder;
use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::arithmetic::ArithmeticConstraint;
use crate::chip::instruction::assign::{AssignInstruction, AssignType};
use crate::chip::instruction::set::AirInstruction;
use crate::chip::register::Register;
use crate::chip::AirParameters;

impl<L: AirParameters> AirBuilder<L> {
    #[inline]
    pub fn assert_expression_zero(&mut self, expression: ArithmeticExpression<L::Field>) {
        let flag = expression.is_trace();
        let constraint = ArithmeticConstraint::All(expression);
        if flag {
            self.constraints.push(constraint.into());
        } else {
            self.global_constraints.push(constraint.into());
        }
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
        let instr = AssignInstruction::new(expression, *data.register(), AssignType::All);
        self.register_air_instruction_internal(AirInstruction::Assign(instr));
    }

    #[inline]
    pub fn set_to_expression_public<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) {
        let instr = AssignInstruction::new(expression, *data.register(), AssignType::All);
        self.register_global_air_instruction_internal(AirInstruction::Assign(instr))
    }

    #[inline]
    pub fn set_to_expression_first_row<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) -> AirInstruction<L::Field, L::Instruction> {
        let instr = AssignInstruction::new(expression, *data.register(), AssignType::First);
        self.register_air_instruction_internal(AirInstruction::Assign(instr.clone()));
        AirInstruction::Assign(instr)
    }

    #[inline]
    pub fn set_to_expression_last_row<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) -> AirInstruction<L::Field, L::Instruction> {
        let instr = AssignInstruction::new(expression, *data.register(), AssignType::Last);
        self.register_air_instruction_internal(AirInstruction::Assign(instr.clone()));
        AirInstruction::Assign(instr)
    }

    #[inline]
    pub fn set_to_expression_transition<T: Register>(
        &mut self,
        data: &T,
        expression: ArithmeticExpression<L::Field>,
    ) -> AirInstruction<L::Field, L::Instruction> {
        let instr = AirInstruction::Assign(AssignInstruction::new(
            expression,
            *data.register(),
            AssignType::Transition,
        ));
        self.register_air_instruction_internal(instr.clone());
        instr
    }

    #[inline]
    pub fn assert_zero(&mut self, data: &impl Register) {
        self.assert_expression_zero(data.expr());
    }

    #[inline]
    pub fn assert_zero_first_row(&mut self, data: &impl Register) {
        self.assert_expression_zero_first_row(data.expr());
    }

    #[inline]
    pub fn assert_zero_last_row(&mut self, data: &impl Register) {
        self.assert_expression_zero_last_row(data.expr());
    }

    #[inline]
    pub fn assert_zero_transition(&mut self, data: &impl Register) {
        self.assert_expression_zero_transition(data.expr());
    }
}
