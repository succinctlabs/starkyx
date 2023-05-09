// use core::marker::PhantomData;

// use plonky2::field::extension::{Extendable, FieldExtension};
// use plonky2::field::packed::PackedField;
// use plonky2::hash::hash_types::RichField;
// use plonky2::plonk::circuit_builder::CircuitBuilder;

// use super::expression::ArithmeticExpression;
// use crate::curta::instruction::Instruction;
// use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};


// #[derive(Debug, Clone)]
// pub enum InstructionConstraint<I : Instruction<F, D>, F: RichField + Extendable<D>, const D: usize> {
//     First(I, PhantomData<F>),
//     Last(I, PhantomData<F>),
//     Transition(I, PhantomData<F>),
//     All(I, PhantomData<F>),
// }

// impl<I : Instruction<F, D>, F: RichField + Extendable<D>, const D: usize> InstructionConstraint<I, F, D> {
//     pub fn packed_generic_constraints<
//         FE,
//         P,
//         const D2: usize,
//         const COLUMNS: usize,
//         const PUBLIC_INPUTS: usize,
//     >(
//         &self,
//         vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
//         yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
//     ) where
//         FE: FieldExtension<D2, BaseField = F>,
//         P: PackedField<Scalar = FE>,
//     {
//         match self {
//             InstructionConstraint::First(instruction, _) => {
//                 let vals = instruction.packed_generic(&vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_first_row(val);
//                 }
//             }
//             InstructionConstraint::Last(instruction, _) => {
//                 let vals = instruction.packed_generic(&vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_last_row(val);
//                 }
//             }
//             InstructionConstraint::Transition(instruction, _) => {
//                 let vals = instruction.packed_generic(&vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_transition(val);
//                 }
//             }
//             InstructionConstraint::All(instruction, _) => {
//                 let vals = instruction.packed_generic(&vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint(val);
//                 }
//             }
//         }
//     }

//     pub fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
//         &self,
//         builder: &mut CircuitBuilder<F, D>,
//         vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
//         yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
//     ) {
//         match self {
//             ArithmeticConstraint::First(constraint) => {
//                 let vals = constraint.expression.ext_circuit(builder, &vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_first_row(builder, val);
//                 }
//             }
//             ArithmeticConstraint::Last(constraint) => {
//                 let vals = constraint.expression.ext_circuit(builder, &vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_last_row(builder, val);
//                 }
//             }
//             ArithmeticConstraint::Transition(constraint) => {
//                 let vals = constraint.expression.ext_circuit(builder, &vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint_transition(builder, val);
//                 }
//             }
//             ArithmeticConstraint::All(constraint) => {
//                 let vals = constraint.expression.ext_circuit(builder, &vars);
//                 for &val in vals.iter() {
//                     yield_constr.constraint(builder, val);
//                 }
//             }
//         }
//     }
// }
