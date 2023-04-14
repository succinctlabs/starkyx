//! An abstraction of Starks for emulated field operations handling all the range_checks
//!
//!
//!

use alloc::collections::BTreeMap;

use anyhow::{anyhow, Result};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::bool::ConstraintBool;
use super::chip::{Chip, ChipParameters};
use super::instruction::arithmetic_expressions::ArithmeticExpression;
use super::instruction::write::WriteInstruction;
use super::instruction::{EqualityConstraint, Instruction};
use super::register::{Array, CellType, MemorySlice, Register, RegisterSerializable};

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum InsID {
    CustomInstruction(Vec<MemorySlice>),
    Write(MemorySlice),
}

#[derive(Clone, Debug)]
pub struct ChipBuilder<L, F, const D: usize>
where
    L: ChipParameters<F, D>,
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

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> Default
    for ChipBuilder<L, F, D>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
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
    pub fn get_local_memory(&mut self, size: usize) -> Result<MemorySlice> {
        let register = MemorySlice::Local(self.local_index, size);
        self.local_index += size;
        if self.local_index > L::NUM_FREE_COLUMNS {
            return Err(anyhow!("Local row memory overflow"));
        }
        Ok(register)
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice` on the
    /// next row.
    fn get_next_memory(&mut self, size: usize) -> Result<MemorySlice> {
        let register = MemorySlice::Next(self.next_index, size);
        self.next_index += size;
        if self.next_index > L::NUM_FREE_COLUMNS {
            return Err(anyhow!("Next row memory overflow"));
        }
        Ok(register)
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice`. Each
    /// cell will be range checked using the lookup table to be in the range `[0, 2^16]`.
    fn get_local_u16_memory(&mut self, size: usize) -> Result<MemorySlice> {
        let register = MemorySlice::Local(self.local_arithmetic_index, size);
        self.local_arithmetic_index += size;
        if self.local_arithmetic_index > L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS {
            return Err(anyhow!("Local row u16 memory overflow"));
        }
        Ok(register)
    }

    /// Allocates `size` cells/columns worth of memory and returns it as a `MemorySlice` in the
    /// next row. Each cell will be range checked using the lookup table to be in the range
    /// `[0, 2^16]`.
    fn get_next_u16_memory(&mut self, size: usize) -> Result<MemorySlice> {
        let register = MemorySlice::Next(self.next_arithmetic_index, size);
        self.local_arithmetic_index += size;
        if self.local_arithmetic_index > L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS {
            return Err(anyhow!("Next row u16 memory overflow"));
        }
        Ok(register)
    }

    /// Allocates a new local register according to type `T` which implements the Register trait
    /// and returns it.
    pub fn alloc_local<T: Register>(&mut self) -> Result<T> {
        let register = match T::CELL {
            Some(CellType::U16) => self.get_local_u16_memory(T::size_of())?,
            Some(CellType::Bit) => {
                let reg = self.get_local_memory(T::size_of())?;
                let consr = EqualityConstraint::<F, D>::Bool(ConstraintBool(reg));
                self.constraints.push(consr);
                reg
            }
            None => self.get_local_memory(T::size_of())?,
        };
        Ok(T::from_register(register))
    }

    pub fn alloc_local_array<T: Register>(&mut self, length: usize) -> Result<Array<T>> {
        let size_of = T::size_of() * length;
        let register = match T::CELL {
            Some(CellType::U16) => self.get_local_u16_memory(size_of)?,
            Some(CellType::Bit) => {
                let reg = self.get_local_memory(size_of)?;
                let consr = EqualityConstraint::Bool(ConstraintBool(reg));
                self.constraints.push(consr);
                reg
            }
            None => self.get_local_memory(size_of)?,
        };
        Ok(Array::<T>::from_register_unsafe(register))
    }

    /// Allocates a new register on the next row according to type `T` which implements the Register
    /// trait and returns it.
    pub fn alloc_next<T: Register>(&mut self) -> Result<T> {
        let register = match T::CELL {
            Some(CellType::U16) => self.get_next_u16_memory(T::size_of())?,
            Some(CellType::Bit) => {
                let reg = self.get_next_memory(T::size_of())?;
                let consr = EqualityConstraint::<F, D>::Bool(ConstraintBool(reg));
                self.constraints.push(consr);
                reg
            }
            None => self.get_next_memory(T::size_of())?,
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
        let id = InsID::CustomInstruction(instruction.memory_vec());
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

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
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
