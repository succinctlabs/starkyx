//! Constraints for instructions
//!
//! The instructions constraints can be multiplied by an arithmetic expression. This is usually
//! used for a selector.

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::expression::ArithmeticExpression;
use crate::curta::instruction::Instruction;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Debug, Clone)]
pub enum InstructionConstraint<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize> {
    First(I, Option<ArithmeticExpression<F, D>>),
    Last(I, Option<ArithmeticExpression<F, D>>),
    Transition(I, Option<ArithmeticExpression<F, D>>),
    All(I, Option<ArithmeticExpression<F, D>>),
}

impl<I: Instruction<F, D>, F: RichField + Extendable<D>, const D: usize>
    InstructionConstraint<I, F, D>
{
    pub fn packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        match self {
            InstructionConstraint::First(instruction, maybe_multiplier) => {
                let vals = instruction.packed_generic(vars);
                match maybe_multiplier {
                    Some(multiplier) => {
                        assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                        let mult = multiplier.expression.packed_generic(vars)[0];
                        for &val in vals.iter() {
                            yield_constr.constraint_first_row(val * mult);
                        }
                    }
                    None => {
                        for &val in vals.iter() {
                            yield_constr.constraint_first_row(val);
                        }
                    }
                }
            }
            InstructionConstraint::Last(instruction, maybe_multiplier) => {
                let vals = instruction.packed_generic(vars);
                match maybe_multiplier {
                    Some(multiplier) => {
                        assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                        let mult = multiplier.expression.packed_generic(vars)[0];
                        for &val in vals.iter() {
                            yield_constr.constraint_last_row(val * mult);
                        }
                    }
                    None => {
                        for &val in vals.iter() {
                            yield_constr.constraint_last_row(val);
                        }
                    }
                }
            }
            InstructionConstraint::Transition(instruction, maybe_multiplier) => {
                let vals = instruction.packed_generic(vars);
                match maybe_multiplier {
                    Some(multiplier) => {
                        assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                        let mult = multiplier.expression.packed_generic(vars)[0];
                        for &val in vals.iter() {
                            yield_constr.constraint_transition(val * mult);
                        }
                    }
                    None => {
                        for &val in vals.iter() {
                            yield_constr.constraint_transition(val);
                        }
                    }
                }
            }
            InstructionConstraint::All(instruction, maybe_multiplier) => {
                let vals = instruction.packed_generic(vars);
                match maybe_multiplier {
                    Some(multiplier) => {
                        assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                        let mult = multiplier.expression.packed_generic(vars)[0];
                        for &val in vals.iter() {
                            yield_constr.constraint(val * mult);
                        }
                    }
                    None => {
                        for &val in vals.iter() {
                            yield_constr.constraint(val);
                        }
                    }
                }
            }
        }
    }

    pub fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            InstructionConstraint::First(instruction, maybe_multiplier) => {
                let vals = instruction.ext_circuit(builder, vars);
                match maybe_multiplier {
                    Some(multiplier) => {
                        assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                        let mult = multiplier.expression.ext_circuit(builder, vars)[0];
                        for &val in vals.iter() {
                            let consr = builder.mul_extension(val, mult);
                            yield_constr.constraint_first_row(builder, consr);
                        }
                    }
                    None => {
                        for &val in vals.iter() {
                            yield_constr.constraint_first_row(builder, val);
                        }
                    }
                }
            }
            InstructionConstraint::Last(instruction, maybe_multiplier) => {
                let vals = instruction.ext_circuit(builder, vars);
                match maybe_multiplier {
                    Some(multiplier) => {
                        assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                        let mult = multiplier.expression.ext_circuit(builder, vars)[0];
                        for &val in vals.iter() {
                            let consr = builder.mul_extension(val, mult);
                            yield_constr.constraint_last_row(builder, consr);
                        }
                    }
                    None => {
                        for &val in vals.iter() {
                            yield_constr.constraint_last_row(builder, val);
                        }
                    }
                }
            }
            InstructionConstraint::Transition(instruction, maybe_multiplier) => {
                let vals = instruction.ext_circuit(builder, vars);
                match maybe_multiplier {
                    Some(multiplier) => {
                        assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                        let mult = multiplier.expression.ext_circuit(builder, vars)[0];
                        for &val in vals.iter() {
                            let consr = builder.mul_extension(val, mult);
                            yield_constr.constraint_transition(builder, consr);
                        }
                    }
                    None => {
                        for &val in vals.iter() {
                            yield_constr.constraint_transition(builder, val);
                        }
                    }
                }
            }
            InstructionConstraint::All(instruction, maybe_multiplier) => {
                let vals = instruction.ext_circuit(builder, vars);
                match maybe_multiplier {
                    Some(multiplier) => {
                        assert_eq!(multiplier.size, 1, "Multiplier must be a single element");
                        let mult = multiplier.expression.ext_circuit(builder, vars)[0];
                        for &val in vals.iter() {
                            let consr = builder.mul_extension(val, mult);
                            yield_constr.constraint(builder, consr);
                        }
                    }
                    None => {
                        for &val in vals.iter() {
                            yield_constr.constraint(builder, val);
                        }
                    }
                }
            }
        }
    }
}
