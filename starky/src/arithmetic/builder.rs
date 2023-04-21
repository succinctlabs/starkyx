//! An abstraction of Starks for emulated field operations handling all the range_checks
//!
//!
//!

use alloc::collections::BTreeMap;

use anyhow::{anyhow, Result};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::bool::ConstraintBool;
use super::chip::{Chip, StarkParameters};
use super::instruction::arithmetic_expressions::ArithmeticExpression;
use super::instruction::write::WriteInstruction;
use super::instruction::{EqualityConstraint, Instruction};
use super::register::{ArrayRegister, CellType, MemorySlice, Register, RegisterSerializable};

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum InsID {
    CustomInstruction(Vec<MemorySlice>),
    Write(MemorySlice),
}

#[derive(Clone, Debug)]
pub struct StarkBuilder<L, F, const D: usize>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    instruction_indices: BTreeMap<InsID, usize>,
    instructions: Vec<L::Instruction>,
    write_instructions: Vec<WriteInstruction>,
    constraints: Vec<EqualityConstraint<F, D>>,
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
            local_index: 0,
            next_index: 0,
            local_arithmetic_index: L::NUM_FREE_COLUMNS,
            next_arithmetic_index: L::NUM_FREE_COLUMNS,
            instruction_indices: BTreeMap::new(),
            instructions: Vec::new(),
            write_instructions: Vec::new(),
            constraints: Vec::new(),
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
                let consr = EqualityConstraint::<F, D>::Bool(ConstraintBool(reg));
                self.constraints.push(consr);
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
                let consr = EqualityConstraint::Bool(ConstraintBool(reg));
                self.constraints.push(consr);
                reg
            }
            None => self.get_local_memory(size_of),
        };
        ArrayRegister::<T>::from_register_unsafe(register)
    }

    /// Allocates a new register on the next row according to type `T` which implements the Register
    /// trait and returns it.
    pub fn alloc_next<T: Register>(&mut self) -> Result<T> {
        let register = match T::CELL {
            Some(CellType::U16) => self.get_next_u16_memory(T::size_of()),
            Some(CellType::Bit) => {
                let reg = self.get_next_memory(T::size_of());
                let consr = EqualityConstraint::<F, D>::Bool(ConstraintBool(reg));
                self.constraints.push(consr);
                reg
            }
            None => self.get_next_memory(T::size_of()),
        };
        Ok(T::from_register(register))
    }

    /// This method should be applied to any data that needs to be manually written to the trace by
    /// the user during trace generation. It currently does not do any actual checks, but this can
    /// be changed later.
    pub fn write_data<T: Register>(&mut self, data: &T) -> Result<()> {
        let register = data.register();
        let label = InsID::Write(*register);
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
        let label = InsID::Write(*register);
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
    pub fn insert_instruction(&mut self, instruction: L::Instruction) -> Result<()> {
        let id = InsID::CustomInstruction(instruction.witness_layout());
        let existing_value = self.instruction_indices.insert(id, self.instructions.len());
        if existing_value.is_some() {
            return Err(anyhow!("Instruction label already exists"));
        }
        self.instructions.push(instruction);
        Ok(())
    }

    /// Asserts that two elements are equal
    pub fn assert_equal<T: Register>(&mut self, a: &T, b: &T) {
        let a = a.register();
        let b = b.register();
        let constraint = EqualityConstraint::Equal(*a, *b);
        self.constraints.push(constraint);
    }

    /// Asserts that two elements are equal
    pub fn insert_raw_constraint(&mut self, constraint: EqualityConstraint<F, D>) {
        self.constraints.push(constraint);
    }

    /// Build the chip
    pub fn build(self) -> (Chip<L, F, D>, BTreeMap<InsID, usize>) {
        let num_free_columns = self.local_index;
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
        let num_arithmetic_columns = self.local_arithmetic_index - self.local_index;
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
                range_checks_idx: (
                    L::NUM_FREE_COLUMNS,
                    L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS,
                ),
                table_index: L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS,
            },
            self.instruction_indices,
        )
    }
}

// Implement methods for the basic operations

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub fn assert_expressions_equal(
        &mut self,
        a: ArithmeticExpression<F, D>,
        b: ArithmeticExpression<F, D>,
    ) {
        let constraint = EqualityConstraint::ArithmeticConstraint(a, b);
        self.constraints.push(constraint);
    }

    /// Asserts that a + b = c
    pub fn add_pointwise<T: Register>(&mut self, a: &T, b: &T, c: &T) {
        let a_exp = ArithmeticExpression::new(a);
        let b_exp = ArithmeticExpression::new(b);
        let c_exp = ArithmeticExpression::new(c);
        self.assert_expressions_equal(a_exp + b_exp, c_exp);
    }

    /// Asserts that a - b = c
    pub fn sub_pointwise<T: Register>(&mut self, a: &T, b: &T, c: &T) {
        let a_exp = ArithmeticExpression::new(a);
        let b_exp = ArithmeticExpression::new(b);
        let c_exp = ArithmeticExpression::new(c);
        self.assert_expressions_equal(a_exp - b_exp, c_exp);
    }

    /// Asserts that a * b = c
    pub fn mul<T: Register>(&mut self, a: &T, b: &T, c: &T) {
        let a_exp = ArithmeticExpression::new(a);
        let b_exp = ArithmeticExpression::new(b);
        let c_exp = ArithmeticExpression::new(c);
        self.assert_expressions_equal(a_exp * b_exp, c_exp);
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
    use crate::arithmetic::chip::TestStark;
    use crate::arithmetic::instruction::DefaultInstructions;
    use crate::arithmetic::register::ElementRegister;
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
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

        type Instruction = DefaultInstructions<F, D>;
    }

    #[test]
    fn test_builder_fibonacchi_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type L = TestChipParameters<F, D>;
        type A = ArithmeticExpression<F, D>;
        type S = TestStark<L, F, D>;

        // event logger to show messages
        let _ = env_logger::builder().is_test(true).try_init();

        let mut builder = StarkBuilder::<L, F, D>::new();

        let x_0 = builder.alloc::<ElementRegister>();
        let x_1 = builder.alloc::<ElementRegister>();
        builder.write_data(&x_0).unwrap();
        builder.write_data(&x_1).unwrap();

        // Asserts that x_0_next = x_1
        builder.assert_equal(&x_0.next(), &x_1);
        // Asserts that x_1_next = x_0 + x_1
        builder.assert_expressions_equal(A::new(&x_0) + A::new(&x_1), A::new(&x_1.next()));

        let (chip, spec) = builder.build();

        let num_rows = 1 << 5;
        let (handle, generator) = trace::<F, D>(spec);

        let mut timing = TimingTree::new("Fibonacchi stark", log::Level::Debug);

        let mut x_0_val = F::ONE;
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
        let stark = TestStark::new(chip);

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
