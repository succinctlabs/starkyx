use alloc::collections::BTreeMap;

use anyhow::{anyhow, Result};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::chip::{Chip, StarkParameters};
use super::constraint::arithmetic::ArithmeticExpression;
use super::constraint::expression::ConstraintExpression;
use super::constraint::Constraint;
use super::instruction::write::WriteInstruction;
use super::instruction::Instruction;
use super::lookup::Lookup;
use super::register::{
    ArrayRegister, CellType, ElementRegister, MemorySlice, Register, RegisterSerializable,
};

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum InstructionId {
    CustomInstruction(Vec<MemorySlice>),
    Write(MemorySlice),
}

pub struct StarkBuilder<L, F, const D: usize>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    instruction_indices: BTreeMap<InstructionId, usize>,
    instructions: Vec<L::Instruction>,
    write_instructions: Vec<WriteInstruction>,
    pub(crate) constraints: Vec<Constraint<L::Instruction, F, D>>,
    pub(crate) range_data: Option<Lookup>,
    pub(crate) range_table: Option<ElementRegister>,
    pub(crate) num_verifier_challenges: usize,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> Default
    for StarkBuilder<L, F, D>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub fn new() -> Self {
        Self {
            local_index: L::NUM_ARITHMETIC_COLUMNS, //0,
            next_index: L::NUM_ARITHMETIC_COLUMNS,  //0,
            local_arithmetic_index: 0,              //L::NUM_FREE_COLUMNS,
            next_arithmetic_index: 0,               //L::NUM_FREE_COLUMNS,
            instruction_indices: BTreeMap::new(),
            instructions: Vec::new(),
            write_instructions: Vec::new(),
            constraints: Vec::new(),
            range_data: None,
            range_table: None,
            num_verifier_challenges: 0,
        }
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice`.
    pub fn get_local_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Local(self.local_index, size);
        self.local_index += size;
        register
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice` on the
    /// next row.
    fn get_next_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Next(self.next_index, size);
        self.next_index += size;
        register
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice`. Each
    /// cell will be range checked using the lookup table to be in the range `[0, 2^16]`.
    fn get_local_u16_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Local(self.local_arithmetic_index, size);
        self.local_arithmetic_index += size;
        register
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice` in the
    /// next row. Each cell will be range checked using the lookup table to be in the range
    /// `[0, 2^16]`.
    fn get_next_u16_memory(&mut self, size: usize) -> MemorySlice {
        let register = MemorySlice::Next(self.next_arithmetic_index, size);
        self.local_arithmetic_index += size;
        register
    }

    /// Allocates a new local register according to type `T` which implements the Register trait
    /// and returns it.
    pub fn alloc<T: Register>(&mut self) -> T {
        let register = match T::CELL {
            Some(CellType::U16) => self.get_local_u16_memory(T::size_of()),
            Some(CellType::Bit) => {
                let reg = self.get_local_memory(T::size_of());
                let constraint_expr =
                    ConstraintExpression::from(reg.expr() * (reg.expr() - F::ONE));
                self.constraints.push(Constraint::All(constraint_expr));
                reg
            }
            None => self.get_local_memory(T::size_of()),
        };
        T::from_register(register)
    }

    pub fn alloc_array<T: Register>(&mut self, length: usize) -> ArrayRegister<T> {
        let size_of = T::size_of() * length;
        let register = match T::CELL {
            Some(CellType::U16) => self.get_local_u16_memory(size_of),
            Some(CellType::Bit) => {
                let reg = self.get_local_memory(size_of);
                let constraint = Constraint::All((reg.expr() * (reg.expr() - F::ONE)).into());
                self.constraints.push(constraint);
                reg
            }
            None => self.get_local_memory(size_of),
        };
        ArrayRegister::<T>::from_register_unsafe(register)
    }

    /// Allocates a new register on the next row according to type `T` which implements the Register
    /// trait and returns it.
    pub fn alloc_next<T: Register>(&mut self) -> T {
        let register = match T::CELL {
            Some(CellType::U16) => self.get_next_u16_memory(T::size_of()),
            Some(CellType::Bit) => {
                let reg = self.get_next_memory(T::size_of());
                let constraint = Constraint::All((reg.expr() * (reg.expr() - F::ONE)).into());
                self.constraints.push(constraint);
                reg
            }
            None => self.get_next_memory(T::size_of()),
        };
        T::from_register(register)
    }

    /// This method should be applied to any data that needs to be manually written to the trace by
    /// the user during trace generation. It currently does not do any actual checks, but this can
    /// be changed later.
    pub fn write_data<T: Register>(&mut self, data: &T) -> Result<()> {
        let register = data.register();
        let label = InstructionId::Write(*register);
        let existing_value = self
            .instruction_indices
            .insert(label, self.write_instructions.len());
        if existing_value.is_some() {
            return Err(anyhow!("Instruction label already exists"));
        }
        self.write_instructions.push(WriteInstruction(*register));
        Ok(())
    }

    /// Inserts a new instruction to the chip
    pub fn write_raw_register(&mut self, data: &MemorySlice) -> Result<()> {
        let register = data;
        let label = InstructionId::Write(*register);
        let existing_value = self
            .instruction_indices
            .insert(label, self.write_instructions.len());
        if existing_value.is_some() {
            return Err(anyhow!("Instruction label already exists"));
        }
        self.write_instructions.push(WriteInstruction(*register));
        Ok(())
    }

    /// Registers a new instruction to the chip.
    fn register_instruction(&mut self, instruction: L::Instruction) -> Result<()> {
        let id = InstructionId::CustomInstruction(instruction.trace_layout());
        let existing_value = self.instruction_indices.insert(id, self.instructions.len());
        if existing_value.is_some() {
            //return Err(anyhow!("Instruction label already exists"));
        }
        self.instructions.push(instruction);
        Ok(())
    }

    /// Registers the instruction to the chip and asserts all its constraints.
    pub fn constrain_instruction(&mut self, instruction: L::Instruction) -> Result<()> {
        self.constraint(instruction.expr())
    }

    pub fn constraint(
        &mut self,
        constraint_expr: ConstraintExpression<L::Instruction, F, D>,
    ) -> Result<()> {
        let instructions = constraint_expr.instructions();
        for instr in instructions {
            self.register_instruction(instr)?;
        }
        let constraint = Constraint::All(constraint_expr);
        self.constraints.push(constraint);
        Ok(())
    }

    /// Asserts that two registers are equal in the first row.
    pub fn assert_equal_first_row<T: Register>(&mut self, a: &T, b: &T) {
        let constraint = Constraint::First((a.expr() - b.expr()).into());
        self.constraints.push(constraint);
    }

    /// Asserts that two registers are equal in the last row.
    pub fn assert_equal_last_row<T: Register>(&mut self, a: &T, b: &T) {
        let constraint = Constraint::Last((a.expr() - b.expr()).into());
        self.constraints.push(constraint);
    }

    /// Asserts that two registers are equal in all rows but the last (useful for when dealing with
    /// registers between the local and next row).
    pub fn assert_equal_transition<T: Register>(&mut self, a: &T, b: &T) {
        let constraint = Constraint::Transition((a.expr() - b.expr()).into());
        self.constraints.push(constraint);
    }

    pub fn assert_equal<T: Register>(&mut self, a: &T, b: &T) {
        let constraint = Constraint::All((a.expr() - b.expr()).into());
        self.constraints.push(constraint);
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in the first row.
    pub fn assert_expressions_equal_first_row(
        &mut self,
        a: ArithmeticExpression<F, D>,
        b: ArithmeticExpression<F, D>,
    ) {
        assert_eq!(a.size, b.size, "Expressions must have the same size");
        let constraint = Constraint::First((a - b).into());
        self.constraints.push(constraint);
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in the last row.
    pub fn assert_expressions_equal_last_row(
        &mut self,
        a: ArithmeticExpression<F, D>,
        b: ArithmeticExpression<F, D>,
    ) {
        assert_eq!(a.size, b.size, "Expressions must have the same size");
        let constraint = Constraint::Last((a - b).into());
        self.constraints.push(constraint);
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in all rows but the last (useful for when dealing with registers between the local and
    /// next row).
    pub fn assert_expressions_equal_transition(
        &mut self,
        a: ArithmeticExpression<F, D>,
        b: ArithmeticExpression<F, D>,
    ) {
        assert_eq!(a.size, b.size, "Expressions must have the same size");
        let constraint = Constraint::Transition((a - b).into());
        self.constraints.push(constraint);
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in all rows.
    pub fn assert_expressions_equal(
        &mut self,
        a: ArithmeticExpression<F, D>,
        b: ArithmeticExpression<F, D>,
    ) {
        assert_eq!(a.size, b.size, "Expressions must have the same size");
        let constraint = Constraint::All((a - b).into());
        self.constraints.push(constraint);
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in the first row.
    pub fn assert_expression_zero_first_row(&mut self, a: ArithmeticExpression<F, D>) {
        let constraint = Constraint::First(a.into());
        self.constraints.push(constraint);
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in the last row.
    pub fn assert_expression_zero_last_row(&mut self, a: ArithmeticExpression<F, D>) {
        let constraint = Constraint::Last(a.into());
        self.constraints.push(constraint);
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in all rows but the last (useful for when dealing with registers between the local and
    /// next row).
    pub fn assert_expression_zero_transition(&mut self, a: ArithmeticExpression<F, D>) {
        let constraint = Constraint::Transition(a.into());
        self.constraints.push(constraint);
    }

    /// Asserts that two expressions (note that expressions are actually vectors of expression) are
    /// equal in all rows.
    pub fn assert_expression_zero(&mut self, a: ArithmeticExpression<F, D>) {
        let constraint = Constraint::All(a.into());
        self.constraints.push(constraint);
    }

    /// Build the chip
    pub fn build(mut self) -> (Chip<L, F, D>, BTreeMap<InstructionId, usize>) {
        let partial_trace_index = self.local_index;

        if L::NUM_ARITHMETIC_COLUMNS > 0 {
            self.arithmetic_range_checks();
        }

        let num_free_columns = self.local_index - L::NUM_ARITHMETIC_COLUMNS; //self.local_index;
        if num_free_columns > L::NUM_FREE_COLUMNS {
            panic!(
                "Not enough free columns. Expected {} free columns, got {}.",
                num_free_columns,
                L::NUM_FREE_COLUMNS
            );
        } else if num_free_columns < L::NUM_FREE_COLUMNS {
            println!(
                "Warning: {} free columns unused",
                L::NUM_FREE_COLUMNS - num_free_columns
            );
        }
        let num_arithmetic_columns = self.local_arithmetic_index; //- self.local_index;
        if num_arithmetic_columns > L::NUM_ARITHMETIC_COLUMNS {
            panic!(
                "Not enough arithmetic columns. Expected {} arithmetic columns, got {}.",
                num_arithmetic_columns,
                L::NUM_ARITHMETIC_COLUMNS
            );
        } else if num_arithmetic_columns < L::NUM_ARITHMETIC_COLUMNS {
            println!(
                "Warning: {} arithmetic columns unused",
                L::NUM_ARITHMETIC_COLUMNS - num_arithmetic_columns
            );
        }
        (
            Chip {
                instructions: self.instructions,
                write_instructions: self.write_instructions,
                constraints: self.constraints,
                range_checks_idx: (0, L::NUM_ARITHMETIC_COLUMNS),
                range_data: self.range_data,
                range_table: self.range_table,
                num_verifier_challenges: self.num_verifier_challenges,
                partial_trace_index,
            },
            self.instruction_indices,
        )
    }
}

// Implement methods for the basic operations
impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    /// Asserts that a + b = c
    pub fn add_pointwise<T: Register>(&mut self, a: &T, b: &T, c: &T) {
        self.assert_expressions_equal(a.expr() + b.expr(), c.expr());
    }

    /// Asserts that a - b = c
    pub fn sub_pointwise<T: Register>(&mut self, a: &T, b: &T, c: &T) {
        self.assert_expressions_equal(a.expr() - b.expr(), c.expr());
    }

    /// Asserts that a * b = c
    pub fn mul<T: Register>(&mut self, a: &T, b: &T, c: &T) {
        self.assert_expressions_equal(a.expr() * b.expr(), c.expr());
    }
}

#[cfg(test)]
mod tests {
    use log::info;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::chip::ChipStark;
    use crate::curta::instruction::EmptyInstructionSet;
    use crate::curta::register::ElementRegister;
    use crate::curta::trace::trace;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Copy, Debug)]
    pub struct TestChipParameters<F, const D: usize> {
        _marker: core::marker::PhantomData<F>,
    }

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D>
        for TestChipParameters<F, D>
    {
        const NUM_FREE_COLUMNS: usize = 2;
        const NUM_ARITHMETIC_COLUMNS: usize = 0;

        type Instruction = EmptyInstructionSet<F, D>;
    }

    #[test]
    fn test_builder_fibonacchi_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type L = TestChipParameters<F, D>;
        type S = ChipStark<L, F, D>;

        let mut builder = StarkBuilder::<L, F, D>::new();
        let x_0 = builder.alloc::<ElementRegister>();
        let x_1 = builder.alloc::<ElementRegister>();
        builder.write_data(&x_0).unwrap();
        builder.write_data(&x_1).unwrap();
        builder.assert_equal_transition(&x_0.next(), &x_1);
        builder.assert_expressions_equal_transition(x_0.expr() + x_1.expr(), x_1.next().expr());
        let (chip, spec) = builder.build();

        let num_rows = 1 << 5;
        let (handle, generator) = trace::<F, D>(spec);
        let mut timing = TimingTree::new("Fibonacchi stark", log::Level::Debug);
        let mut x_0_val = F::ZERO;
        let mut x_1_val = F::ONE;
        let trace = timed!(timing, "generate trace", {
            for i in 0..num_rows {
                handle.write_data(i, x_0, vec![x_0_val]).unwrap();
                handle.write_data(i, x_1, vec![x_1_val]).unwrap();
                if i == num_rows - 1 {
                    break;
                }
                (x_0_val, x_1_val) = (x_1_val, x_0_val + x_1_val);
            }
            drop(handle);
            generator.generate_trace(&chip, num_rows as usize).unwrap()
        });
        info!("The 32 fibonacchi numbers is {:?}", x_0_val);
        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip);

        // Verify proof as a stark
        let proof = timed!(
            timing,
            "generate stark proof",
            prove::<F, C, S, D>(
                stark.clone(),
                &config,
                trace,
                [],
                &mut TimingTree::default(),
            )
            .unwrap()
        );
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof = add_virtual_stark_proof_with_pis(
            &mut recursive_builder,
            stark.clone(),
            &config,
            degree_bits,
        );

        recursive_builder.print_gate_counts(0);

        let mut rec_pw = PartialWitness::new();
        set_stark_proof_with_pis_target(&mut rec_pw, &virtual_proof, &proof);

        verify_stark_proof_circuit::<F, C, S, D>(
            &mut recursive_builder,
            stark,
            virtual_proof,
            &config,
        );

        let recursive_data = recursive_builder.build::<C>();

        let recursive_proof = timed!(
            timing,
            "generate recursive proof",
            plonky2::plonk::prover::prove(
                &recursive_data.prover_only,
                &recursive_data.common,
                rec_pw,
                &mut TimingTree::default(),
            )
            .unwrap()
        );

        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }
}
