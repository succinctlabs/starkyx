//! A chip for emulating field operations
//!
//! This chip handles the range checks for the limbs, allocating table columns for input, output,
//! and witness values.
//!

use core::fmt::Debug;
use core::ops::Range;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::constraint::Constraint;
use super::instruction::write::WriteInstruction;
use super::instruction::Instruction;
use super::lookup::Lookup;
use super::register::ElementRegister;
// use crate::lookup::{eval_lookups, eval_lookups_circuit};
// use crate::permutation::PermutationPair;
use crate::stark::Stark;
use crate::curta::new_stark::Stark as NewStark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

/// A layout for a circuit that emulates field operations
pub trait StarkParameters<F: RichField + Extendable<D>, const D: usize>:
    Sized + Send + Sync + Clone
{
    /// The number of columns that need to be ranged-checked to range 0..num_rows
    ///
    /// If NUM_ARITHMETIC_COLUMNS > 0 is used for field operations with 2^16 bit limbs
    /// the number of rows should be 2^16.
    const NUM_ARITHMETIC_COLUMNS: usize;

    /// The number of columns that are not range checked.
    const NUM_FREE_COLUMNS: usize;

    /// Th
    type Instruction: Instruction<F, D> + Debug;
}

#[derive(Debug, Clone)]
pub struct Chip<L, F, const D: usize>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub(crate) instructions: Vec<L::Instruction>,
    pub(crate) write_instructions: Vec<WriteInstruction>,
    pub(crate) constraints: Vec<Constraint<L::Instruction, F, D>>,
    pub(crate) range_checks_idx: (usize, usize),
    pub(crate) range_data: Option<Lookup>,
    pub(crate) range_table: Option<ElementRegister>,
    pub(crate) partial_trace_index: usize,
    pub(crate) num_verifier_challenges: usize,
}

impl<L, F, const D: usize> Chip<L, F, D>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    // #[inline]
    // pub const fn table_index(&self) -> usize {
    //     self.table_index
    // }

    // #[inline]
    // pub const fn relative_table_index(&self) -> usize {
    //     self.table_index
    // }

    #[inline]
    pub const fn range_checks_idx(&self) -> (usize, usize) {
        self.range_checks_idx
    }

    #[inline]
    pub const fn num_columns_no_range_checks(&self) -> usize {
        L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS
    }

    #[inline]
    pub const fn num_range_checks(&self) -> usize {
        L::NUM_ARITHMETIC_COLUMNS
    }

    // #[inline]
    // pub const fn col_perm_index(&self, i: usize) -> usize {
    //     2 * (i - self.range_checks_idx.0) + self.table_index + 1
    // }

    // #[inline]
    // pub const fn table_perm_index(&self, i: usize) -> usize {
    //     2 * (i - self.range_checks_idx.0) + 1 + self.table_index + 1
    // }

    #[inline]
    pub const fn num_columns() -> usize {
        //1 + L::NUM_FREE_COLUMNS + 3 * L::NUM_ARITHMETIC_COLUMNS
        L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS
    }
    #[inline]
    pub const fn arithmetic_range(&self) -> Range<usize> {
        self.range_checks_idx.0..self.range_checks_idx.1
    }

    #[inline]
    pub const fn permutations_range(&self) -> Range<usize> {
        L::NUM_FREE_COLUMNS..Self::num_columns()
    }

    pub fn eval_packed_generic<
        FE,
        P,
        const D2: usize,
        const COLUMNS: usize,
        const PUBLIC_INPUTS: usize,
    >(
        &self,
        betas: &[F],
        vars: StarkEvaluationVars<FE, P, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        for consr in self.constraints.iter() {
            consr.packed_generic_constraints(vars, yield_constr);
        }
        if let Some(range_data) = &self.range_data {
            range_data.packed_generic_constraints::<F, D, FE, P, D2, COLUMNS, PUBLIC_INPUTS>(
                betas,
                vars,
                yield_constr,
            );
        }
        // // lookp table values
        // yield_constr.constraint_first_row(vars.local_values[self.table_index]);
        // let table_values_relation =
        //     vars.local_values[self.table_index] + FE::ONE - vars.next_values[self.table_index];
        // yield_constr.constraint_transition(table_values_relation);
        // // permutations
        // for i in self.arithmetic_range() {
        //     eval_lookups(
        //         vars,
        //         yield_constr,
        //         self.col_perm_index(i),
        //         self.table_perm_index(i),
        //     );
        // }
    }

    pub fn eval_ext_circuit<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
        &self,
        betas: &[Target],
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        for consr in self.constraints.iter() {
            consr.ext_circuit_constraints(builder, vars, yield_constr);
        }
        if let Some(range_data) = &self.range_data {
            range_data.ext_circuit_constraints::<F, D, COLUMNS, PUBLIC_INPUTS>(
                betas,
                builder,
                vars,
                yield_constr,
            );
        }
        // // lookup table values
        // yield_constr.constraint_first_row(builder, vars.local_values[self.table_index]);
        // let one = builder.constant_extension(F::Extension::ONE);
        // let table_plus_one = builder.add_extension(vars.local_values[self.table_index], one);
        // let table_relation =
        //     builder.sub_extension(table_plus_one, vars.next_values[self.table_index]);
        // yield_constr.constraint_transition(builder, table_relation);

        // // lookup argumment
        // for i in self.arithmetic_range() {
        //     eval_lookups_circuit(
        //         builder,
        //         vars,
        //         yield_constr,
        //         self.col_perm_index(i),
        //         self.table_perm_index(i),
        //     );
        // }
    }

    fn constraint_degree(&self) -> usize {
        3
    }
    // fn permutation_pairs(&self) -> Vec<PermutationPair> {
    //     self.arithmetic_range()
    //         .flat_map(|i| {
    //             [
    //                 PermutationPair::singletons(i, self.col_perm_index(i)),
    //                 PermutationPair::singletons(self.table_index, self.table_perm_index(i)),
    //             ]
    //         })
    //         .collect()
    // }
}

/// A Stark for emulated field operations
///
/// This stark handles the range checks for the limbs
#[derive(Debug, Clone)]
pub struct ChipStark<L, F, const D: usize>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub chip: Chip<L, F, D>,
}

impl<L, F, const D: usize> ChipStark<L, F, D>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub fn new(chip: Chip<L, F, D>) -> Self {
        Self { chip }
    }

    #[inline]
    pub fn partial_trace_index(&self) -> usize {
        self.chip.partial_trace_index
    }

    #[inline]
    pub fn num_verifier_challenges(&self) -> usize {
        self.chip.num_verifier_challenges
    }

    #[inline]
    pub fn chip(&self) -> &Chip<L, F, D> {
        &self.chip
    }
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> Stark<F, D>
    for ChipStark<L, F, D>
{
    const COLUMNS: usize = Chip::<L, F, D>::num_columns();
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        self.chip.eval_packed_generic(&vec![], vars, yield_constr)
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        self.chip
            .eval_ext_circuit(&vec![], builder, vars, yield_constr)
    }

    fn constraint_degree(&self) -> usize {
        self.chip.constraint_degree()
    }

    // fn permutation_pairs(&self) -> Vec<PermutationPair> {
    //     self.chip.permutation_pairs()
    // }
}





// impl<L :StarkParameters<F, D> ,F: RichField + Extendable<D>, const D: usize> NewStark<F, D, 2> for ChipStark<L, F, D> {

// }