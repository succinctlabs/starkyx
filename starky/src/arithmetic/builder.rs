//! An abstraction of Starks for emulated field operations handling all the range_checks

use alloc::collections::BTreeMap;

use anyhow::{anyhow, Result};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::chip::{Chip, ChipParameters};
use super::instruction::{Instruction, WriteInstruction};
use super::register::{CellType, DataRegister, Register};

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum InsID {
    Label(String),
    MemID(Vec<Register>),
    Write(Register),
    WriteLabel(String),
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
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> Default
    for ChipBuilder<L, F, D>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
    // pub fn build(self) -> CompliedStark<L, F, D>

    pub fn new() -> Self {
        Self {
            local_index: 0,
            next_index: 0,
            local_arithmetic_index: L::NUM_FREE_COLUMNS,
            next_arithmetic_index: L::NUM_FREE_COLUMNS,
            instruction_indices: BTreeMap::new(),
            instructions: Vec::new(),
            write_instructions: Vec::new(),
        }
    }

    fn get_instruction_from_label(&self, label: &str) -> Option<&L::Instruction> {
        self.instruction_indices
            .get(&InsID::Label(String::from(label)))
            .map(|index| &self.instructions[*index])
    }

    fn get_local_memory(&mut self, size: usize) -> Result<Register> {
        let register = Register::Local(self.local_index, size);
        self.local_index += size;
        if self.local_index > L::NUM_FREE_COLUMNS {
            return Err(anyhow!("Local row memory overflow"));
        }
        Ok(register)
    }

    fn get_next_memory(&mut self, size: usize) -> Result<Register> {
        let register = Register::Next(self.next_index, size);
        self.next_index += size;
        if self.next_index > L::NUM_FREE_COLUMNS {
            return Err(anyhow!("Next row memory overflow"));
        }
        Ok(register)
    }

    fn get_local_u16_memory(&mut self, size: usize) -> Result<Register> {
        let register = Register::Local(self.local_arithmetic_index, size);
        self.local_arithmetic_index += size;
        if self.local_arithmetic_index > L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS {
            return Err(anyhow!("Local row u16 memory overflow"));
        }
        Ok(register)
    }

    fn get_next_u16_memory(&mut self, size: usize) -> Result<Register> {
        let register = Register::Next(self.next_arithmetic_index, size);
        self.local_arithmetic_index += size;
        if self.local_arithmetic_index > L::NUM_ARITHMETIC_COLUMNS + L::NUM_FREE_COLUMNS {
            return Err(anyhow!("Next row u16 memory overflow"));
        }
        Ok(register)
    }

    /// Allocates a new local row register and returns it
    pub fn alloc_local<T: DataRegister>(&mut self) -> Result<T> {
        let register = match T::CELL {
            Some(CellType::U16) => self.get_local_u16_memory(T::size_of())?,
            Some(CellType::Bit) => {
                unimplemented!("Bit cells are not supported yet");
                //self.get_local_memory(T::size_of())?
            }
            None => self.get_local_memory(T::size_of())?,
        };
        Ok(T::from_raw_register(register))
    }

    /// Allocates a new next row register and returns it
    pub fn alloc_next<T: DataRegister>(&mut self) -> Result<T> {
        let register = match T::CELL {
            Some(CellType::U16) => self.get_next_u16_memory(T::size_of())?,
            Some(CellType::Bit) => {
                unimplemented!("Bit cells are not supported yet");
                //self.get_next_memory(T::size_of())?
            }
            None => self.get_next_memory(T::size_of())?,
        };
        Ok(T::from_raw_register(register))
    }

    /// Inserts a new instruction to the chip
    pub fn write_data<T: DataRegister>(&mut self, data: &T) -> Result<()> {
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
    pub fn write_labeled_data<T: DataRegister>(&mut self, data: &T, label: &str) -> Result<()> {
        let register = data.register();
        let label = InsID::WriteLabel(String::from(label));
        let existing_value = self
            .instruction_indices
            .insert(label, self.write_instructions.len());
        if existing_value.is_some() {
            return Err(anyhow!("Instruction label already exists"));
        }
        self.write_instructions.push(WriteInstruction(*register));
        Ok(())
    }

    pub fn insert_instruction(&mut self, instruction: L::Instruction) -> Result<()> {
        let id = InsID::MemID(instruction.memory_vec());
        let existing_value = self.instruction_indices.insert(id, self.instructions.len());
        if existing_value.is_some() {
            return Err(anyhow!("Instruction label already exists"));
        }
        self.insert_raw_instruction(instruction)
    }

    /// An instruction identified by its label
    pub fn insert_labeled_instruction(
        &mut self,
        instruction: L::Instruction,
        label: &str,
    ) -> Result<()> {
        let id = InsID::Label(String::from(label));
        let existing_value = self.instruction_indices.insert(id, self.instructions.len());
        if existing_value.is_some() {
            return Err(anyhow!("Instruction label already exists"));
        }
        self.insert_raw_instruction(instruction)
    }

    fn insert_raw_instruction(&mut self, instruction: L::Instruction) -> Result<()> {
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

    /// Build the chip
    pub fn build(self) -> (Chip<L, F, D>, BTreeMap<InsID, usize>) {
        (
            Chip {
                instructions: self.instructions,
                write_instructions: self.write_instructions,
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

#[cfg(test)]
mod tests {

    use num::bigint::RandBigInt;
    use num::BigUint;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;

    use super::*;
    use crate::arithmetic::chip::TestStark;
    use crate::arithmetic::modular::add::{
        AddModInstruction, NUM_ARITH_COLUMNS as NUM_ADD_COLUMNS, N_LIMBS as NUM_LIMBS,
    };
    use crate::arithmetic::modular::mul::{
        MulModInstruction, NUM_ARITH_COLUMNS as NUM_MUL_COLUMNS,
    };
    use crate::arithmetic::register::U16Array;
    use crate::arithmetic::trace::trace;
    use crate::arithmetic::ArithmeticParser;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Copy, Debug)]
    pub struct AddModTestParameters;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for AddModTestParameters {
        const NUM_ARITHMETIC_COLUMNS: usize = NUM_ADD_COLUMNS;
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = AddModInstruction;
    }

    #[test]
    fn test_builder_add() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type U256 = U16Array<NUM_LIMBS>;
        type S = TestStark<AddModTestParameters, F, D>;
        type L = AddModTestParameters;

        // Build the stark
        let mut builder = ChipBuilder::<AddModTestParameters, F, D>::new();

        let a = builder.alloc_local::<U256>().unwrap();
        let b = builder.alloc_local::<U256>().unwrap();
        let out = builder.alloc_local::<U256>().unwrap();
        let m = builder.alloc_local::<U256>().unwrap();

        let add_instruction = AddModInstruction::new(a, b, m, out);
        builder.insert_instruction(add_instruction).unwrap();

        let (chip, spec) = builder.build();

        let (handle, generator) = trace::<F, D>(spec);

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        for i in 0..num_rows {
            let a = rng.gen_biguint(256) % &p22519;
            let b = rng.gen_biguint(256) % &p22519;
            let m = p22519.clone();
            let h = handle.clone();
            rayon::spawn(move || {
                let row = ArithmeticParser::<F, D>::add_trace(a, b, m);
                h.write(i as usize, add_instruction, row).unwrap();
            });
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows as usize).unwrap();
        let stark = TestStark { chip };

        // Verify proof as a stark
        let proof = prove::<F, C, S, D>(
            stark.clone(),
            &config,
            trace,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
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

        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut timing,
        )
        .unwrap();

        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }

    #[derive(Clone, Copy, Debug)]
    pub struct MulModTestParameters;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for MulModTestParameters {
        const NUM_ARITHMETIC_COLUMNS: usize = NUM_MUL_COLUMNS;
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = MulModInstruction;
    }

    #[test]
    fn test_builder_mul() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Element = U16Array<NUM_LIMBS>;
        type S = TestStark<MulModTestParameters, F, D>;
        type L = MulModTestParameters;

        // Build the stark
        let mut builder = ChipBuilder::<MulModTestParameters, F, D>::new();

        let a = builder
            .alloc_local::<Element>()
            .unwrap()
            .into_raw_register();
        let b = builder
            .alloc_local::<Element>()
            .unwrap()
            .into_raw_register();
        let out = builder
            .alloc_local::<Element>()
            .unwrap()
            .into_raw_register();
        let m = builder
            .alloc_local::<Element>()
            .unwrap()
            .into_raw_register();

        let mul_instruction = MulModInstruction::new(a, b, m, out);
        builder.insert_instruction(mul_instruction).unwrap();

        let (chip, spec) = builder.build();

        let (handle, generator) = trace::<F, D>(spec);

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        for i in 0..num_rows {
            let a = rng.gen_biguint(256) % &p22519;
            let b = rng.gen_biguint(256) % &p22519;
            let m = p22519.clone();
            let h = handle.clone();
            rayon::spawn(move || {
                let row = ArithmeticParser::<F, D>::mul_trace(a, b, m);
                h.write(i as usize, mul_instruction, row).unwrap();
            });
        }

        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows as usize).unwrap();
        let stark = TestStark { chip };

        // Verify proof as a stark
        let proof = prove::<F, C, S, D>(
            stark.clone(),
            &config,
            trace,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
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

        let mut timing = TimingTree::new("recursive_proof", log::Level::Debug);
        let recursive_proof = plonky2::plonk::prover::prove(
            &recursive_data.prover_only,
            &recursive_data.common,
            rec_pw,
            &mut timing,
        )
        .unwrap();

        timing.print();
        recursive_data.verify(recursive_proof).unwrap();
    }
}
