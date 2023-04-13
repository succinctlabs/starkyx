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
use super::instruction::write::WriteInstruction;
use super::instruction::{EqualityConstraint, Instruction};
use super::register::{CellType, MemorySlice, Register};

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
        Ok(T::from_raw_register(register))
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
        Ok(T::from_raw_register(register))
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

        let mut inst = instruction;
        if let Some(data) = inst.witness_data() {
            let (size, cell_type) = data.destruct();
            let register = match cell_type {
                Some(CellType::U16) => self.get_local_u16_memory(size)?,
                Some(CellType::Bit) => {
                    unimplemented!("Bit cells are not supported yet");
                    //self.get_local_memory(size)?
                }
                None => self.get_local_memory(size)?,
            };

            inst.set_witness(register)?;
        }
        self.instructions.push(inst);
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
    pub fn insert_raw_constraint(&mut self, constraint : EqualityConstraint<F, D>) {
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
    // pub fn add<T: Register>(&mut self, a: T, b: T, output: T) {
    //     let a = a.register();
    //     let b = b.register();
    //     let output = output.register();
    //     let instruction = StandardInstruction::Add(*a, *b, *output);
    //     self.insert_standard_instruction(instruction).unwrap();
    // }

    // pub fn sub<T: Register>(&mut self, a: T, b: T, output: T) {
    //     let a = a.register();
    //     let b = b.register();
    //     let output = output.register();
    //     let instruction = StandardInstruction::Sub(*a, *b, *output);
    //     self.insert_standard_instruction(instruction).unwrap();
    // }

    // pub fn mul<T: Register>(&mut self, a: T, b: T, output: T) {
    //     let a = a.register();
    //     let b = b.register();
    //     let output = output.register();
    //     let instruction = StandardInstruction::Mul(*a, *b, *output);
    //     self.insert_standard_instruction(instruction).unwrap();
    // }
}

// #[cfg(test)]
// mod tests {
//     use plonky2::field::types::Sample;
//     use plonky2::iop::witness::PartialWitness;
//     use plonky2::plonk::circuit_builder::CircuitBuilder;
//     use plonky2::plonk::circuit_data::CircuitConfig;
//     use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
//     use plonky2::util::timing::TimingTree;

//     use super::*;
//     use crate::arithmetic::builder::ChipBuilder;
//     use crate::arithmetic::chip::{ChipParameters, TestStark};
//     use crate::arithmetic::field::mul::FpMul;
//     use crate::arithmetic::field::Fp25519Param;
//     use crate::arithmetic::register::ElementRegister;
//     use crate::arithmetic::trace::trace;
//     use crate::config::StarkConfig;
//     use crate::prover::prove;
//     use crate::recursive_verifier::{
//         add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
//         verify_stark_proof_circuit,
//     };
//     use crate::verifier::verify_stark_proof;

//     #[derive(Clone, Debug, Copy)]
//     struct AssertEqualTest;

//     impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for AssertEqualTest {
//         const NUM_ARITHMETIC_COLUMNS: usize = 0;
//         const NUM_FREE_COLUMNS: usize = 3 * 16;

//         type Instruction = FpMul<Fp25519Param>;
//     }

//     #[test]
//     fn test_builder_basic() {
//         const D: usize = 2;
//         type C = PoseidonGoldilocksConfig;
//         type F = <C as GenericConfig<D>>::F;
//         type S = TestStark<AssertEqualTest, F, D>;

//         // build the stark
//         let mut builder = ChipBuilder::<AssertEqualTest, F, D>::new();

//         let a = builder.alloc_local::<ElementRegister>().unwrap();
//         let b = builder.alloc_local::<ElementRegister>().unwrap();
//         let c = builder.alloc_local::<ElementRegister>().unwrap();
//         builder.write_data(&a).unwrap();
//         builder.write_data(&b).unwrap();
//         builder.write_data(&c).unwrap();

//         builder.mul(a, b, c);

//         let (chip, spec) = builder.build();

//         // Construct the trace
//         let num_rows = 2u64.pow(16) as usize;
//         let (handle, generator) = trace::<F, D>(spec);

//         for i in 0..num_rows {
//             let a_val = F::rand();
//             let b_val = F::rand();
//             let c_val = a_val * b_val;
//             handle.write_data(i, a, vec![a_val]).unwrap();
//             handle.write_data(i, b, vec![b_val]).unwrap();
//             handle.write_data(i, c, vec![c_val]).unwrap();
//         }
//         drop(handle);

//         let trace = generator.generate_trace(&chip, num_rows).unwrap();

//         let config = StarkConfig::standard_fast_config();
//         let stark = TestStark::new(chip);

//         // Verify proof as a stark
//         let proof = prove::<F, C, S, D>(
//             stark.clone(),
//             &config,
//             trace,
//             [],
//             &mut TimingTree::default(),
//         )
//         .unwrap();
//         verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

//         // Verify recursive proof in a circuit
//         let config_rec = CircuitConfig::standard_recursion_config();
//         let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

//         let degree_bits = proof.proof.recover_degree_bits(&config);
//         let virtual_proof = add_virtual_stark_proof_with_pis(
//             &mut recursive_builder,
//             stark.clone(),
//             &config,
//             degree_bits,
//         );

//         recursive_builder.print_gate_counts(0);

//         let mut rec_pw = PartialWitness::new();
//         set_stark_proof_with_pis_target(&mut rec_pw, &virtual_proof, &proof);

//         verify_stark_proof_circuit::<F, C, S, D>(
//             &mut recursive_builder,
//             stark,
//             virtual_proof,
//             &config,
//         );

//         let recursive_data = recursive_builder.build::<C>();

//         let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
//         let recursive_proof = plonky2::plonk::prover::prove(
//             &recursive_data.prover_only,
//             &recursive_data.common,
//             rec_pw,
//             &mut timing,
//         )
//         .unwrap();

//         timing.print();
//         recursive_data.verify(recursive_proof).unwrap();
//     }
// }
