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
use super::instruction::{EqualityConstraint, Instruction, WriteInstruction};
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
    constraints: Vec<EqualityConstraint>,
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
            constraints: Vec::new(),
        }
    }

    fn _get_instruction_from_label(&self, label: &str) -> Option<&L::Instruction> {
        self.instruction_indices
            .get(&InsID::Label(String::from(label)))
            .map(|index| &self.instructions[*index])
    }

    pub fn get_local_memory(&mut self, size: usize) -> Result<Register> {
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
                let reg = self.get_local_memory(T::size_of())?;
                let consr = EqualityConstraint::Bool(ConstraintBool(reg));
                self.constraints.push(consr);
                reg
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
                let reg = self.get_next_memory(T::size_of())?;
                let consr = EqualityConstraint::Bool(ConstraintBool(reg));
                self.constraints.push(consr);
                reg
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

    /// Registers a new instruction to the chip.
    pub fn insert_instruction(&mut self, instruction: L::Instruction) -> Result<()> {
        let id = InsID::MemID(instruction.memory_vec());
        let existing_value = self.instruction_indices.insert(id, self.instructions.len());
        if existing_value.is_some() {
            return Err(anyhow!("Instruction label already exists"));
        }
        self.insert_raw_instruction(instruction)
    }

    /// Registers a new instruction identified by a user-specified label
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

    // This function should be made private in the future
    pub fn assert_registers_equal(&mut self, a: &Register, b: &Register) {
        let constraint = EqualityConstraint::Equal(*a, *b);
        self.constraints.push(constraint);
    }

    /// Asserts that two elements are equal
    pub fn assert_equal<T: DataRegister>(&mut self, a: &T, b: &T) {
        let a = a.register();
        let b = b.register();
        self.assert_registers_equal(a, b)
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

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    //use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::arithmetic::builder::ChipBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
    use crate::arithmetic::field::mul::FpMul;
    use crate::arithmetic::field::{FieldParameters, Fp25519, Fp25519Param};
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    struct AssertEqualTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for AssertEqualTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 2 * 16;
        const NUM_FREE_COLUMNS: usize = 1;

        type Instruction = FpMul<Fp25519Param>;
    }

    #[test]
    fn test_builder_assert_equal() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Fp = Fp25519;
        type S = TestStark<AssertEqualTest, F, D>;

        // build the stark
        let mut builder = ChipBuilder::<AssertEqualTest, F, D>::new();

        let a = builder.alloc_local::<Fp>().unwrap();
        let b = builder.alloc_local::<Fp>().unwrap();
        builder.write_data(&a).unwrap();
        builder.write_data(&b).unwrap();

        builder.assert_equal(&a, &b);

        let (chip, spec) = builder.build();

        // Construct the trace
        let num_rows = 2u64.pow(16) as usize;
        let (handle, generator) = trace::<F, D>(spec);

        let p = Fp25519Param::modulus_biguint();

        let mut rng = thread_rng();
        for i in 0..num_rows {
            let a_int = rng.gen_biguint(256) % &p;
            //let handle = handle.clone();
            //rayon::spawn(move || {
            handle.write_field(i, &a_int, a).unwrap();
            handle.write_field(i, &a_int, b).unwrap();
            //});
        }
        drop(handle);

        let trace = generator.generate_trace(&chip, num_rows).unwrap();

        let config = StarkConfig::standard_fast_config();
        let stark = TestStark::new(chip);

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
