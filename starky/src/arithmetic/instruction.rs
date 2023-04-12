//! Instruction trait
//!
//! The instruction trait represents the interface of a microcomand in a chip. It is the
//! lowest level of abstraction in the arithmetic module.

use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::bool::ConstraintBool;
use super::register::{Register, WitnessData};
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub trait Instruction<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Clone
{
    //fn generate_trace_row(&self, input: Option<Self::Input>) -> (Vec<F>, Option<Self::Output>);

    fn memory_vec(&self) -> Vec<Register>;

    fn witness_data(&self) -> Option<WitnessData>;

    fn set_witness(&mut self, witness: Register) -> Result<()>;

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize);

    fn packed_generic_constraints<
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
        P: PackedField<Scalar = FE>;

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    );
}

/// Standard instructions that are included in every instantiation of the builder
///
/// This code might change to be more generic in the future
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StandardInstruction<F, const D: usize> {
    Add(Register, Register, Register),
    AddConst(Register, F, Register),
    Sub(Register, Register, Register),
    SubConst(Register, F, Register),
    Mul(Register, Register, Register),
    MulConst(Register, F, Register),
}

impl<F, const D: usize> StandardInstruction<F, D> {
    pub fn includes_next(&self) -> bool {
        match self {
            StandardInstruction::Add(
                Register::Local(_, _),
                Register::Local(_, _),
                Register::Local(_, _),
            ) => true,
            StandardInstruction::AddConst(Register::Local(_, _), _, Register::Local(_, _)) => true,
            StandardInstruction::Sub(
                Register::Local(_, _),
                Register::Local(_, _),
                Register::Local(_, _),
            ) => true,
            StandardInstruction::SubConst(Register::Local(_, _), _, Register::Local(_, _)) => true,
            StandardInstruction::Mul(
                Register::Local(_, _),
                Register::Local(_, _),
                Register::Local(_, _),
            ) => true,
            StandardInstruction::MulConst(Register::Local(_, _), _, Register::Local(_, _)) => true,
            _ => false,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for StandardInstruction<F, D> {
    fn memory_vec(&self) -> Vec<Register> {
        match self {
            StandardInstruction::Add(a, b, c) => vec![*a, *b, *c],
            StandardInstruction::AddConst(a, _, c) => vec![*a, *c],
            StandardInstruction::Sub(a, b, c) => vec![*a, *b, *c],
            StandardInstruction::SubConst(a, _, c) => vec![*a, *c],
            StandardInstruction::Mul(a, b, c) => vec![*a, *b, *c],
            StandardInstruction::MulConst(a, _, c) => vec![*a, *c],
        }
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            StandardInstruction::Add(_, _, c) => c.assign(trace_rows, row, row_index),
            StandardInstruction::AddConst(_, _, c) => c.assign(trace_rows, row, row_index),
            StandardInstruction::Sub(_, _, c) => c.assign(trace_rows, row, row_index),
            StandardInstruction::SubConst(_, _, c) => c.assign(trace_rows, row, row_index),
            StandardInstruction::Mul(_, _, c) => c.assign(trace_rows, row, row_index),
            StandardInstruction::MulConst(_, _, c) => c.assign(trace_rows, row, row_index),
        }
    }

    fn set_witness(&mut self, witness: Register) -> Result<()> {
        Ok(())
    }

    fn witness_data(&self) -> Option<WitnessData> {
        None
    }

    fn packed_generic_constraints<
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
        let includes_next = self.includes_next();
        match self {
            StandardInstruction::Add(a, b, c) => {
                let a_vals = a.packed_entries_slice(&vars);
                let b_vals = b.packed_entries_slice(&vars);
                let c_vals = c.packed_entries_slice(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(b_vals.iter())
                    .zip(c_vals.iter())
                    .map(|((a, b), c)| *a + *b - *c);

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(consr);
                    }
                }
            }

            StandardInstruction::AddConst(a, b, c) => {
                let a_vals = a.packed_entries_slice(&vars);
                let c_vals = c.packed_entries_slice(&vars);
                let scalar = FE::from_basefield(*b);

                let constraints = a_vals
                    .iter()
                    .zip(c_vals.iter())
                    .map(|(a, c)| *a + scalar - *c);

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(consr);
                    }
                }
            }

            StandardInstruction::Sub(a, b, c) => {
                let a_vals = a.packed_entries_slice(&vars);
                let b_vals = b.packed_entries_slice(&vars);
                let c_vals = c.packed_entries_slice(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(b_vals.iter())
                    .zip(c_vals.iter())
                    .map(|((a, b), c)| *a - *b - *c);

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(consr);
                    }
                }
            }
            StandardInstruction::SubConst(a, b, c) => {
                let a_vals = a.packed_entries_slice(&vars);
                let c_vals = c.packed_entries_slice(&vars);
                let scalar = FE::from_basefield(*b);

                let constraints = a_vals
                    .iter()
                    .zip(c_vals.iter())
                    .map(|(a, c)| *a - scalar - *c);

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(consr);
                    }
                }
            }
            StandardInstruction::Mul(a, b, c) => {
                let a_vals = a.packed_entries_slice(&vars);
                let b_vals = b.packed_entries_slice(&vars);
                let c_vals = c.packed_entries_slice(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(b_vals.iter())
                    .zip(c_vals.iter())
                    .map(|((a, b), c)| *a * *b - *c);

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(consr);
                    }
                }
            }
            StandardInstruction::MulConst(a, b, c) => {
                let a_vals = a.packed_entries_slice(&vars);
                let c_vals = c.packed_entries_slice(&vars);
                let scalar = FE::from_basefield(*b);

                let constraints = a_vals
                    .iter()
                    .zip(c_vals.iter())
                    .map(|(a, c)| *a * scalar - *c);

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(consr);
                    }
                }
            }
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        let includes_next = self.includes_next();
        match self {
            StandardInstruction::Add(a, b, c) => {
                let a_vals = a.evaluation_targets(&vars);
                let b_vals = b.evaluation_targets(&vars);
                let c_vals = c.evaluation_targets(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(b_vals.iter())
                    .zip(c_vals.iter())
                    .map(|((a, b), c)| {
                        let c_exp = builder.add_extension(*a, *b);
                        builder.sub_extension(*c, c_exp)
                    })
                    .collect::<Vec<_>>();

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(builder, consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(builder, consr);
                    }
                }
            }
            StandardInstruction::AddConst(a, b, c) => {
                let a_vals = a.evaluation_targets(&vars);
                let c_vals = c.evaluation_targets(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(c_vals.iter())
                    .map(|(a, c)| {
                        let c_exp = builder.add_const_extension(*a, *b);
                        builder.sub_extension(*c, c_exp)
                    })
                    .collect::<Vec<_>>();

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(builder, consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(builder, consr);
                    }
                }
            }
            StandardInstruction::Sub(a, b, c) => {
                let a_vals = a.evaluation_targets(&vars);
                let b_vals = b.evaluation_targets(&vars);
                let c_vals = c.evaluation_targets(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(b_vals.iter())
                    .zip(c_vals.iter())
                    .map(|((a, b), c)| {
                        let c_exp = builder.sub_extension(*a, *b);
                        builder.sub_extension(*c, c_exp)
                    })
                    .collect::<Vec<_>>();

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(builder, consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(builder, consr);
                    }
                }
            }
            StandardInstruction::SubConst(a, b, c) => {
                let a_vals = a.evaluation_targets(&vars);
                let c_vals = c.evaluation_targets(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(c_vals.iter())
                    .map(|(a, c)| {
                        let b_neg = -(*b);
                        let c_exp = builder.add_const_extension(*a, b_neg);
                        builder.sub_extension(*c, c_exp)
                    })
                    .collect::<Vec<_>>();

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(builder, consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(builder, consr);
                    }
                }
            }

            StandardInstruction::Mul(a, b, c) => {
                let a_vals = a.evaluation_targets(&vars);
                let b_vals = b.evaluation_targets(&vars);
                let c_vals = c.evaluation_targets(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(b_vals.iter())
                    .zip(c_vals.iter())
                    .map(|((a, b), c)| {
                        let c_exp = builder.mul_extension(*a, *b);
                        builder.sub_extension(*c, c_exp)
                    })
                    .collect::<Vec<_>>();

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(builder, consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(builder, consr);
                    }
                }
            }
            StandardInstruction::MulConst(a, b, c) => {
                let a_vals = a.evaluation_targets(&vars);
                let c_vals = c.evaluation_targets(&vars);

                let constraints = a_vals
                    .iter()
                    .zip(c_vals.iter())
                    .map(|(a, c)| {
                        let c_exp = builder.mul_const_extension(*b, *a);
                        builder.sub_extension(*c, c_exp)
                    })
                    .collect::<Vec<_>>();

                if includes_next {
                    for consr in constraints {
                        yield_constr.constraint_transition(builder, consr);
                    }
                } else {
                    for consr in constraints {
                        yield_constr.constraint(builder, consr);
                    }
                }
            }
        }
    }
}

/// Constraints don't have any writing data or witness
///
/// These are only used to generate constrains and can therefore
/// be held separate from instructions and all hold the same type.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EqualityConstraint {
    Bool(ConstraintBool),
    Equal(Register, Register),
}

#[derive(Clone, Debug, Copy)]
pub struct WriteInstruction(pub Register);

impl WriteInstruction {
    #[inline]
    pub fn into_register(self) -> Register {
        self.0
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for WriteInstruction {
    fn witness_data(&self) -> Option<WitnessData> {
        None
    }

    fn memory_vec(&self) -> Vec<Register> {
        vec![self.0]
    }

    fn set_witness(&mut self, _witness: Register) -> Result<()> {
        Ok(())
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        self.0.assign(trace_rows, row, row_index);
    }

    fn packed_generic_constraints<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        _vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        _yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        _yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for EqualityConstraint {
    fn witness_data(&self) -> Option<WitnessData> {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::witness_data(constraint)
            }
            EqualityConstraint::Equal(_a, _b) => None,
        }
    }

    fn memory_vec(&self) -> Vec<Register> {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::memory_vec(constraint)
            }
            EqualityConstraint::Equal(_a, _b) => vec![],
        }
    }

    fn set_witness(&mut self, _witness: Register) -> Result<()> {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::set_witness(constraint, _witness)
            }
            EqualityConstraint::Equal(_a, _b) => Ok(()),
        }
    }

    fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::assign_row(
                    constraint, trace_rows, row, row_index,
                )
            }
            EqualityConstraint::Equal(_, _) => {}
        }
    }

    fn packed_generic_constraints<
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
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::packed_generic_constraints(
                    constraint,
                    vars,
                    yield_constr,
                )
            }
            EqualityConstraint::Equal(a, b) => {
                let a_vals = a.packed_entries_slice(&vars);
                let b_vals = b.packed_entries_slice(&vars);
                if let (Register::Local(_, _), Register::Local(_, _)) = (a, b) {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        yield_constr.constraint(a - b);
                    }
                } else {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        yield_constr.constraint_transition(a - b);
                    }
                }
            }
        }
    }

    fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        match self {
            EqualityConstraint::Bool(constraint) => {
                <ConstraintBool as Instruction<F, D>>::ext_circuit_constraints(
                    constraint,
                    builder,
                    vars,
                    yield_constr,
                )
            }
            EqualityConstraint::Equal(a, b) => {
                let a_vals = a.evaluation_targets(&vars);
                let b_vals = b.evaluation_targets(&vars);
                if let (Register::Local(_, _), Register::Local(_, _)) = (a, b) {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        let constr = builder.sub_extension(a, b);
                        yield_constr.constraint(builder, constr);
                    }
                } else {
                    for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
                        let constr = builder.sub_extension(a, b);
                        yield_constr.constraint_transition(builder, constr);
                    }
                }
            }
        }
    }
}
