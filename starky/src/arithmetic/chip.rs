use core::ops::Range;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::instruction::{Instruction, WriteInstruction};
use crate::lookup::{eval_lookups, eval_lookups_circuit};
use crate::permutation::PermutationPair;
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

/// A layout for a circuit that emulates field operations
pub trait ChipParameters<F: RichField + Extendable<D>, const D: usize>:
    Sized + Send + Sync
{
    const NUM_ARITHMETIC_COLUMNS: usize;
    const NUM_FREE_COLUMNS: usize;

    type Instruction: Instruction<F, D>;
}

#[derive(Debug, Clone)]
pub struct Chip<L, F, const D: usize>
where
    L: ChipParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub(crate) instructions: Vec<L::Instruction>,
    pub(crate) write_instructions: Vec<WriteInstruction>,
    pub(crate) range_checks_idx: (usize, usize),
    pub(crate) table_index: usize,
}

impl<L, F, const D: usize> Chip<L, F, D>
where
    L: ChipParameters<F, D>,
    F: RichField + Extendable<D>,
{
    #[inline]
    pub const fn table_index(&self) -> usize {
        self.table_index
    }

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

    #[inline]
    pub const fn col_perm_index(&self, i: usize) -> usize {
        2 * i + self.table_index + 1
    }

    #[inline]
    pub const fn table_perm_index(&self, i: usize) -> usize {
        2 * i + 1 + self.table_index + 1
    }

    #[inline]
    pub const fn num_columns() -> usize {
        1 + L::NUM_FREE_COLUMNS + 3 * L::NUM_ARITHMETIC_COLUMNS
    }
    #[inline]
    pub const fn arithmetic_range(&self) -> Range<usize> {
        self.range_checks_idx.0..self.range_checks_idx.1
    }
}

/// A Stark for emulated field operations
///
/// This stark handles the range checks for the limbs
#[derive(Clone)]
pub struct TestStark<L, F, const D: usize>
where
    L: ChipParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub(crate) chip: Chip<L, F, D>,
}

impl<L, F, const D: usize> TestStark<L, F, D>
where
    L: ChipParameters<F, D>,
    F: RichField + Extendable<D>,
{
    pub fn new(chip: Chip<L, F, D>) -> Self {
        Self { chip }
    }
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> Stark<F, D>
    for TestStark<L, F, D>
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
        for layout in self.chip.instructions.iter() {
            layout.packed_generic_constraints(vars, yield_constr);
        }
        // lookp table values
        yield_constr.constraint_first_row(vars.local_values[self.chip.table_index]);
        let table_values_relation = vars.local_values[self.chip.table_index] + FE::ONE
            - vars.next_values[self.chip.table_index];
        yield_constr.constraint_transition(table_values_relation);
        // permutations
        for i in self.chip.arithmetic_range() {
            eval_lookups(
                vars,
                yield_constr,
                self.chip.col_perm_index(i),
                self.chip.table_perm_index(i),
            );
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
        for layout in self.chip.instructions.iter() {
            layout.ext_circuit_constraints(builder, vars, yield_constr);
        }
        // lookup table values
        yield_constr.constraint_first_row(builder, vars.local_values[self.chip.table_index]);
        let one = builder.constant_extension(F::Extension::ONE);
        let table_plus_one = builder.add_extension(vars.local_values[self.chip.table_index], one);
        let table_relation =
            builder.sub_extension(table_plus_one, vars.next_values[self.chip.table_index]);
        yield_constr.constraint_transition(builder, table_relation);

        // lookup argumment
        for i in self.chip.arithmetic_range() {
            eval_lookups_circuit(
                builder,
                vars,
                yield_constr,
                self.chip.col_perm_index(i),
                self.chip.table_perm_index(i),
            );
        }
    }

    fn constraint_degree(&self) -> usize {
        2
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        self.chip
            .arithmetic_range()
            .flat_map(|i| {
                [
                    PermutationPair::singletons(i, self.chip.col_perm_index(i)),
                    PermutationPair::singletons(
                        self.chip.table_index,
                        self.chip.table_perm_index(i),
                    ),
                ]
            })
            .collect()
    }
}
