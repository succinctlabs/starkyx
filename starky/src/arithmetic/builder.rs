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

    /// Inserts a new subchip to the chip
    ///
    /// Input:
    ///     - chip: The subchip to insert
    ///    - instruction_indices: A map from the instruction labels to the instruction indices in the subchip
    /// Returns:
    /// .   - Returns an error if the instruction label already exists of if there is not enough memory
    pub fn insert_chip<S: ChipParameters<F, D>>(
        &mut self,
        chip: Chip<S, F, D>,
        instruction_indices: BTreeMap<InsID, usize>,
    ) -> Result<Chip<S, F, D>>
    where
        S::Instruction: Into<L::Instruction>,
    {
        let mut sub_chip = chip;
        let length = self.instructions.len();
        let free_shift = std::cmp::max(self.local_index, self.next_index);
        let arithmetic_shift =
            std::cmp::max(self.local_arithmetic_index, self.next_arithmetic_index);
        self.local_index += S::NUM_FREE_COLUMNS;
        self.next_index += S::NUM_FREE_COLUMNS;
        self.local_arithmetic_index += S::NUM_ARITHMETIC_COLUMNS;
        self.next_arithmetic_index += S::NUM_ARITHMETIC_COLUMNS;
        for (id, index) in instruction_indices {
            // Shift all the instructions to their new location

            let new_id = match id {
                InsID::MemID(_) => {
                    let instruction = &mut sub_chip.instructions[index];
                    instruction.shift_right(free_shift, arithmetic_shift);
                    InsID::MemID(instruction.memory_vec())
                }
                InsID::Label(label) => {
                    let instruction = &mut sub_chip.instructions[index];
                    instruction.shift_right(free_shift, arithmetic_shift);
                    InsID::Label(label)
                }
                InsID::Write(_) => {
                    let instruction = &mut sub_chip.write_instructions[index];
                    <WriteInstruction as Instruction<F, D>>::shift_right(
                        instruction,
                        free_shift,
                        arithmetic_shift,
                    );
                    InsID::Write(instruction.into_register())
                }
                InsID::WriteLabel(label) => {
                    let instruction = &mut sub_chip.write_instructions[index];
                    <WriteInstruction as Instruction<F, D>>::shift_right(
                        instruction,
                        free_shift,
                        arithmetic_shift,
                    );
                    InsID::WriteLabel(label)
                }
            };
            // Insert the instruction index to the chip map
            let existing_value = self.instruction_indices.insert(new_id, length + index);
            if existing_value.is_some() {
                return Err(anyhow!("Conflicting instructions"));
            }
        }
        // let new_sub_chip = Chip {
        //     instructions: chip_i,
        //     write_instructions: sub_chip.write_instructions,
        //     range_checks_idx: (
        //         length + S::NUM_FREE_COLUMNS,
        //         length + S::NUM_FREE_COLUMNS + S::NUM_ARITHMETIC_COLUMNS,
        //     ),
        //     table_index: L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS,
        // };
        self.instructions
            .extend(sub_chip.instructions.clone().into_iter().map(Into::into));
        self.write_instructions
            .extend(sub_chip.write_instructions.clone().into_iter());
        Ok(sub_chip)
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
    use plonky2::field::extension::FieldExtension;
    use plonky2::field::packed::PackedField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;

    use super::*;
    use crate::arithmetic::chip::TestStark;
    use crate::arithmetic::modular::add::{
        AddModChip, AddModInstruction, NUM_ARITH_COLUMNS as NUM_ADD_COLUMNS, N_LIMBS as NUM_LIMBS,
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
    use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
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

    #[derive(Clone, Copy, Debug)]
    pub enum ModInstruction {
        Add(AddModInstruction),
        Mul(MulModInstruction),
    }

    impl<F: RichField + Extendable<D>, const D: usize> Instruction<F, D> for ModInstruction {
        fn assign_row(&self, trace_rows: &mut [Vec<F>], row: &mut [F], row_index: usize) {
            match self {
                ModInstruction::Add(add) => add.assign_row(trace_rows, row, row_index),
                ModInstruction::Mul(mul) => mul.assign_row(trace_rows, row, row_index),
            }
        }

        fn witness_data(&self) -> Option<crate::arithmetic::register::WitnessData> {
            match self {
                ModInstruction::Add(add) => {
                    <AddModInstruction as Instruction<F, D>>::witness_data(add)
                }
                ModInstruction::Mul(mul) => {
                    <MulModInstruction as Instruction<F, D>>::witness_data(mul)
                }
            }
        }

        fn memory_vec(&self) -> Vec<Register> {
            match self {
                ModInstruction::Add(add) => {
                    <AddModInstruction as Instruction<F, D>>::memory_vec(add)
                }
                ModInstruction::Mul(mul) => {
                    <MulModInstruction as Instruction<F, D>>::memory_vec(mul)
                }
            }
        }

        fn set_witness(&mut self, witness: Register) -> Result<()> {
            match self {
                ModInstruction::Add(add) => {
                    <AddModInstruction as Instruction<F, D>>::set_witness(add, witness)
                }
                ModInstruction::Mul(mul) => {
                    <MulModInstruction as Instruction<F, D>>::set_witness(mul, witness)
                }
            }
        }

        fn shift_right(&mut self, free_shift: usize, arithmetic_shift: usize) {
            match self {
                ModInstruction::Add(add) => <AddModInstruction as Instruction<F, D>>::shift_right(
                    add,
                    free_shift,
                    arithmetic_shift,
                ),
                ModInstruction::Mul(mul) => <MulModInstruction as Instruction<F, D>>::shift_right(
                    mul,
                    free_shift,
                    arithmetic_shift,
                ),
            }
        }

        fn ext_circuit_constraints<const COLUMNS: usize, const PUBLIC_INPUTS: usize>(
            &self,
            builder: &mut CircuitBuilder<F, D>,
            vars: StarkEvaluationTargets<D, { COLUMNS }, { PUBLIC_INPUTS }>,
            yield_constr: &mut crate::constraint_consumer::RecursiveConstraintConsumer<F, D>,
        ) {
            match self {
                ModInstruction::Add(add) => {
                    <AddModInstruction as Instruction<F, D>>::ext_circuit_constraints(
                        add,
                        builder,
                        vars,
                        yield_constr,
                    )
                }
                ModInstruction::Mul(mul) => {
                    <MulModInstruction as Instruction<F, D>>::ext_circuit_constraints(
                        mul,
                        builder,
                        vars,
                        yield_constr,
                    )
                }
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
                ModInstruction::Add(add) => {
                    <AddModInstruction as Instruction<F, D>>::packed_generic_constraints(
                        add,
                        vars,
                        yield_constr,
                    )
                }
                ModInstruction::Mul(mul) => {
                    <MulModInstruction as Instruction<F, D>>::packed_generic_constraints(
                        mul,
                        vars,
                        yield_constr,
                    )
                }
            }
        }
    }

    impl From<AddModInstruction> for ModInstruction {
        fn from(add: AddModInstruction) -> Self {
            ModInstruction::Add(add)
        }
    }

    impl From<MulModInstruction> for ModInstruction {
        fn from(mul: MulModInstruction) -> Self {
            ModInstruction::Mul(mul)
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct ModTestParameters;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for ModTestParameters {
        const NUM_ARITHMETIC_COLUMNS: usize = NUM_ADD_COLUMNS + NUM_MUL_COLUMNS;
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = ModInstruction;
    }

    #[test]
    fn test_add_mul_chip() {
        // make the add chip
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type U256 = U16Array<NUM_LIMBS>;
        type L = ModTestParameters;
        type S = TestStark<L, F, D>;
        type Element = U16Array<NUM_LIMBS>;

        type LAdd = AddModTestParameters;
        type LMul = MulModTestParameters;

        // Build the add chip
        let mut add_builder = ChipBuilder::<LAdd, F, D>::new();

        let a = add_builder.alloc_local::<U256>().unwrap();
        let b = add_builder.alloc_local::<U256>().unwrap();
        let out = add_builder.alloc_local::<U256>().unwrap();
        let m = add_builder.alloc_local::<U256>().unwrap();

        let add_instruction = AddModInstruction::new(a, b, m, out);
        add_builder
            .insert_instruction(add_instruction.into())
            .unwrap();

        let (add_chip, add_spec) = AddModChip::new();

        // Biuld the mul chip
        let mut mul_builder = ChipBuilder::<MulModTestParameters, F, D>::new();

        let a = mul_builder
            .alloc_local::<Element>()
            .unwrap()
            .into_raw_register();
        let b = mul_builder
            .alloc_local::<Element>()
            .unwrap()
            .into_raw_register();
        let out = mul_builder
            .alloc_local::<Element>()
            .unwrap()
            .into_raw_register();
        let m = mul_builder
            .alloc_local::<Element>()
            .unwrap()
            .into_raw_register();

        let mul_instruction = MulModInstruction::new(a, b, m, out);
        mul_builder.insert_instruction(mul_instruction).unwrap();

        let (mul_chip, mul_spec) = mul_builder.build();

        // Build the main chip

        let mut builder = ChipBuilder::<L, F, D>::new();
        let sub_add_chip = builder.insert_chip(add_chip, add_spec).unwrap();
        let sub_add_inst = sub_add_chip.instructions[0];
        let sub_mul_chip = builder.insert_chip(mul_chip, mul_spec).unwrap();
        let sub_mul_inst = sub_mul_chip.instructions[0];

        let (chip, spec) = builder.build();
        assert_eq!(chip.instructions.len(), 2);

        let (handle, generator) = trace::<F, D>(spec);

        let a_mul = <MulModInstruction as Instruction<F, D>>::memory_vec(&sub_mul_inst)[0];
        assert_eq!(a_mul, Register::Local(NUM_ADD_COLUMNS, NUM_LIMBS));
        let b_mul = <MulModInstruction as Instruction<F, D>>::memory_vec(&sub_mul_inst)[1];
        assert_eq!(
            b_mul,
            Register::Local(NUM_ADD_COLUMNS + NUM_LIMBS, NUM_LIMBS)
        );
        let out_mul = <MulModInstruction as Instruction<F, D>>::memory_vec(&sub_mul_inst)[3];
        assert_eq!(
            out_mul,
            Register::Local(NUM_ADD_COLUMNS + 2 * NUM_LIMBS, NUM_LIMBS)
        );
        let m_mul = <MulModInstruction as Instruction<F, D>>::memory_vec(&sub_mul_inst)[2];
        assert_eq!(
            m_mul,
            Register::Local(NUM_ADD_COLUMNS + 3 * NUM_LIMBS, NUM_LIMBS)
        );
        let carry = sub_mul_inst.carry.unwrap();
        assert_eq!(
            carry,
            Register::Local(NUM_ADD_COLUMNS + 4 * NUM_LIMBS, NUM_LIMBS)
        );

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        for i in 0..num_rows {
            let a = rng.gen_biguint(256) % &p22519;
            let b = rng.gen_biguint(256) % &p22519;
            let m = p22519.clone();

            let a2 = rng.gen_biguint(256) % &p22519;
            let b2 = rng.gen_biguint(256) % &p22519;
            let m2 = p22519.clone();

            let h = handle.clone();
            rayon::spawn(move || {
            let add_row = ArithmeticParser::<F, D>::add_trace(a, b, m);
            h.write(i as usize, sub_add_inst, add_row).unwrap();
            let mul_row = ArithmeticParser::<F, D>::mul_trace(a2, b2, m2);
            h.write(i as usize, sub_mul_inst, mul_row).unwrap();
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
