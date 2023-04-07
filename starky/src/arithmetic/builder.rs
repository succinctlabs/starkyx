//! An abstraction of Starks for emulated field operations handling all the range_checks

use alloc::collections::BTreeMap;
use core::ops::Range;
use std::sync::mpsc;

use anyhow::{anyhow, Result};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::util::transpose;
use plonky2_maybe_rayon::*;

use super::instruction::{Instruction, InstructionID, LabeledInstruction};
use super::register::{CellType, DataRegister, WitnessData};
use super::Register;
use super::trace::{TraceHandle, TraceGenerator};
use crate::arithmetic::circuit::{ChipParameters, StarkParameters};
use crate::lookup::{eval_lookups, eval_lookups_circuit, permuted_cols};
use crate::permutation::PermutationPair;
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};

pub struct ChipBuilder<L, F, const D: usize>
where
    L: ChipParameters<F, D>,
    F: RichField + Extendable<D>,
{
    local_index: usize,
    local_arithmetic_index: usize,
    next_arithmetic_index: usize,
    next_index: usize,
    instruction_indices: BTreeMap<InstructionID, usize>,
    instructions: Vec<L::Instruction>,
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
        }
    }

    fn get_instruction(&self, id: InstructionID) -> Option<&L::Instruction> {
        self.instruction_indices
            .get(&id)
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
        self.local_index += size;
        if self.local_index > L::NUM_ARITHMETIC_COLUMNS {
            return Err(anyhow!("Local row u16 memory overflow"));
        }
        Ok(register)
    }

    fn get_next_u16_memory(&mut self, size: usize) -> Result<Register> {
        let register = Register::Next(self.next_arithmetic_index, size);
        self.next_index += size;
        if self.next_index > L::NUM_ARITHMETIC_COLUMNS {
            return Err(anyhow!("Next row u16 memory overflow"));
        }
        Ok(register)
    }

    fn register_cell_type(&mut self, cell_type: CellType, register: Register) -> Result<()> {
        match cell_type {
            CellType::U16 => {}
            CellType::Bit => unimplemented!(),
        }
        Ok(())
    }

    /// Allocates a new local row register and returns it
    pub fn alloc_local<T: DataRegister>(&mut self) -> Result<T> {
        let register = match T::CELL {
            Some(CellType::U16) => self.get_local_u16_memory(T::size_of())?,
            Some(CellType::Bit) => {
                unimplemented!("Bit cells are not supported yet");
                self.get_local_memory(T::size_of())?
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
                self.get_next_memory(T::size_of())?
            }
            None => self.get_next_memory(T::size_of())?,
        };
        Ok(T::from_raw_register(register))
    }

    /// Inserts a new instruction to the chip
    pub fn insert_instruction<T>(&mut self, instruction: LabeledInstruction<T, F, D>) -> Result<()>
    where
        T: Into<L::Instruction> + Instruction<F, D>,
    {
        let (label, mut instruction) = instruction.destruct();
        let existing_value = self
            .instruction_indices
            .insert(label, self.instructions.len());
        if existing_value.is_some() {
            return Err(anyhow!("Instruction label already exists"));
        }
        let (size, cell_type) = instruction.witness_data().destruct();
        let register = match cell_type {
            Some(CellType::U16) => self.get_next_u16_memory(size)?,
            Some(CellType::Bit) => {
                unimplemented!("Bit cells are not supported yet");
                self.get_next_memory(size)?
            }
            None => self.get_next_memory(size)?,}; 
        
        instruction.set_witness(register)?;
        self.instructions.push(instruction.into());

        Ok(())
    }

    /// Inserts a new subchip to the chip
    /// 
    /// Input: 
    ///     - chip: The subchip to insert
    ///    - instruction_indices: A map from the instruction labels to the instruction indices in the subchip
    /// Returns an error if the instruction label already exists of if there is not enough memory
    pub fn insert_chip<S: StarkParameters<F, D>>(&mut self, mut chip: Chip<S, F, D>, instruction_indices: BTreeMap<InstructionID, usize>,) -> Result<()>
    where
        S::Instruction: Into<L::Instruction>,
    {

        let length = self.instructions.len();
        let free_shift = std::cmp::max(self.local_index, self.next_index);
        let arithmetic_shift = std::cmp::max(self.local_arithmetic_index, self.next_arithmetic_index);
        for (id, index) in instruction_indices {
            // Insert the instruction index to the chip map
            let existing_value = self.instruction_indices.insert(id, length + index);
            // Shift all the instructions to their new location
            chip.instructions[index].shift_right(free_shift, arithmetic_shift);
            if existing_value.is_some() {
                return Err(anyhow!("Instruction label already exists"));
            }
        }
        let chip_instructions = chip.instructions.into_iter().map(Into::into);
        self.instructions.extend(chip_instructions);
        Ok(())
    }

    /// Building the Stark
    pub fn build(self) -> Chip<L, F, D> {
        Chip {
            instructions: self.instructions,
            range_checks_idx: (
                self.local_arithmetic_index,
                self.local_arithmetic_index + L::NUM_ARITHMETIC_COLUMNS,
            ),
            table_index: self.local_arithmetic_index + L::NUM_ARITHMETIC_COLUMNS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chip<L, F, const D: usize>
where
    L: ChipParameters<F, D>,
    F: RichField + Extendable<D>,
{
    instructions: Vec<L::Instruction>,
    range_checks_idx: (usize, usize),
    table_index: usize,
}

impl<L, F, const D: usize> Chip<L, F, D>
where
    L: ChipParameters<F, D>,
    F: RichField + Extendable<D>, {

        #[inline]
        pub const fn table_index(&self) -> usize {
            self.table_index
        }

        #[inline]
        pub const fn num_columns_no_range_checks(&self) -> usize {
            L::NUM_FREE_COLUMNS + L::NUM_ARITHMETIC_COLUMNS
        }

        #[inline]
        pub const fn num_range_checks(&self) -> usize {
            self.range_checks_idx.1 - self.range_checks_idx.0
        }

        #[inline]
        pub const fn col_perm_index(&self, i: usize) -> usize {
            2 * i + self.table_index+ 1
        }

        #[inline]
        pub const fn table_perm_index(&self, i: usize) -> usize {
            2 * i + 1 + self.table_index + 1
        }

        #[inline]
        pub const fn num_columns() -> usize {
            1 + L::NUM_FREE_COLUMNS +3 * L::NUM_ARITHMETIC_COLUMNS
        }
        #[inline]
        pub const fn arithmetic_range(&self) -> Range<usize> {
            self.range_checks_idx.0..self.range_checks_idx.1
        }

        pub fn get(&self, index: usize) -> &L::Instruction {
            &self.instructions[index]
        }

    }

/// A Stark for emulated field operations
///
/// This stark handles the range checks for the limbs
#[derive(Clone)]
pub struct TestStark<L, F, const D: usize>
where
    L: StarkParameters<F, D>,
    F: RichField + Extendable<D>,
{
    chip : Chip<L, F, D>,
}


impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize>
    TestStark<L, F, D>
{
    /*     /// Generate the trace for the arithmetic circuit
    pub fn generate_trace(
        &self,
        program: Vec<impl InstructionT<L, F, D>>,
    ) -> Vec<PolynomialValues<F>> {
        let num_operations = program.len();
        let num_rows = num_operations;

        let mut trace_rows = vec![vec![F::ZERO; L::NUM_ARITHMETIC_COLUMNS + 1]; num_rows];

        // Collecte the trace rows which are processed in parallel
        let (tx, rx) = mpsc::channel();

        for (pc, instruction) in program.into_iter().enumerate() {
            let tx = tx.clone();
            instruction.generate_trace(pc, tx);
        }
        drop(tx);

        // Insert the trace rows into the trace
        while let Ok((i, op_index, mut row)) = rx.recv() {
            L::OPERATIONS[op_index].assign_row(&mut trace_rows, &mut row, i)
        }

        // Transpose the trace to get the columns and resize to the correct size
        let mut trace_cols = transpose(&trace_rows);
        trace_cols.resize(Self::num_columns(), Vec::with_capacity(num_rows));

        trace_cols[L::TABLE_INDEX]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                *x = F::from_canonical_usize(i);
            });

        // Calculate the permutation and append permuted columbs to trace
        let (trace_values, perm_values) = trace_cols.split_at_mut(L::TABLE_INDEX + 1);
        (0..L::NUM_ARITHMETIC_COLUMNS)
            .into_par_iter()
            .map(|i| permuted_cols(&trace_values[i], &trace_values[L::TABLE_INDEX]))
            .zip(perm_values.par_iter_mut().chunks(2))
            .for_each(|((col_perm, table_perm), mut trace)| {
                trace[0].extend(col_perm);
                trace[1].extend(table_perm);
            });

        trace_cols
            .into_par_iter()
            .map(PolynomialValues::new)
            .collect()
    } */
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> Stark<F, D>
    for TestStark<L, F, D>
{
    const COLUMNS: usize = Chip::<L, F, D>::num_columns();
    const PUBLIC_INPUTS: usize = L::PUBLIC_INPUTS;

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
        let table_values_relation =
            vars.local_values[self.chip.table_index] + FE::ONE - vars.next_values[self.chip.table_index];
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
        self.chip.arithmetic_range()
            .flat_map(|i| {
                [
                    PermutationPair::singletons(i, self.chip.col_perm_index(i)),
                    PermutationPair::singletons(self.chip.table_index, self.chip.table_perm_index(i)),
                ]
            })
            .collect()
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

    use super::*;
    use crate::arithmetic::layout::Opcode;
    use crate::arithmetic::modular::add::{
        AddModInstruction, NUM_ARITH_COLUMNS as NUM_ADD_COLUMNS, N_LIMBS as NUM_ADD_LIMBS,
    };
    use crate::arithmetic::register::U16Array;
    use crate::arithmetic::Register;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Copy, Debug)]
    pub struct AddTestParameters;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for AddTestParameters {
        const NUM_ARITHMETIC_COLUMNS: usize = NUM_ADD_COLUMNS;
        const PUBLIC_INPUTS: usize = 0;
        const NUM_STARK_COLUMNS: usize = 0;

        type Instruction = AddModInstruction;
    }

    #[derive(Clone, Copy, Debug)]
    pub struct AddTrace;

    //impl<F: RichField + Extendable<D>, const D: usize> TraceBuilder<AddTestParameters, F, D> for AddTrace {
/* 
        fn generate_trace(&self, _ :(), writer: &TraceWriter<AddTestParameters, F, D>)  {
            let mut rng = rand::thread_rng();
            let num_rows =  100;
            let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

            for i in 0..num_rows {
                let a = rng.gen_biguint(256) % &p22519;
                let b = rng.gen_biguint(256) % &p22519;
                //writer.execute(i, InstructionID("addmod".into()), (a, b, p22519.clone()));
            }

        }
    } */

    #[test]
    fn test_builder_add() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Element = U16Array<NUM_ADD_LIMBS>;

        // Build the stark
        let mut builder = ChipBuilder::<AddTestParameters, F, D>::new();

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

        let add_instruction = AddModInstruction::new(a, b, m, out);
        let add_labeled =
            LabeledInstruction::new(InstructionID(String::from("add")), add_instruction);

        builder.insert_instruction(add_labeled).unwrap();

        let chip = builder.build();
        let stark = TestStark{chip};

        // Construct the trace
        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        for i in 0..num_rows {
            let a = rng.gen_biguint(256) % &p22519;
            let b = rng.gen_biguint(256) % &p22519; 
        }
    }

    //impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for AddTestParameters {
    //    const NUM_ARITHMETIC_COLUMNS: usize = add::NUM_ARITH_COLUMNS;
    //    const PUBLIC_INPUTS: usize = 0;
    //    const NUM_STARK_COLUMNS: usize = 0;
    //}

    /*   impl<F: RichField + Extendable<D>, const D: usize> EmulatedCircuitLayout<F, D, 1>
        for AddModLayoutCircuit
    {
        const PUBLIC_INPUTS: usize = 0;
        const NUM_ARITHMETIC_COLUMNS: usize = add::NUM_ARITH_COLUMNS;
        const ENTRY_COLUMN: usize = 0;
        const TABLE_INDEX: usize = add::NUM_ARITH_COLUMNS;

        type Layouts = ArithmeticLayout;
        const OPERATIONS: [ArithmeticLayout; 1] = [ArithmeticLayout::Add(add::AddModLayout::new(
            Register::Local(0, add::N_LIMBS),
            Register::Local(add::N_LIMBS, add::N_LIMBS),
            Register::Local(2 * add::N_LIMBS, add::N_LIMBS),
            Register::Local(3 * add::N_LIMBS, add::N_LIMBS),
            Register::Local(4 * add::N_LIMBS, add::NUM_ADD_WITNESS_COLUMNS),
        ))];
    } */
    /*
    #[derive(Debug, Clone)]
    pub struct AddInstruction {
        pub a: BigUint,
        pub b: BigUint,
        pub modulus: BigUint,
    }

    impl AddInstruction {
        pub fn new(a: BigUint, b: BigUint, modulus: BigUint) -> Self {
            Self { a, b, modulus }
        }
    }

    impl<F: RichField + Extendable<D>, const D: usize> InstructionT<AddModLayoutCircuit, F, D, 1>
        for AddInstruction
    {
        fn generate_trace(self, pc: usize, tx: mpsc::Sender<(usize, usize, Vec<F>)>) {
            let operation = ArithmeticOp::AddMod(self.a, self.b, self.modulus);
            let (trace_row, _) = operation.generate_trace_row();
            tx.send((pc, 0, trace_row)).unwrap();
        }
    }

    #[test]
    fn test_arithmetic_stark_add() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = CompliedStark<AddModLayoutCircuit, 1, F, D>;

        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        let mut additions = Vec::new();

        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(255) % &p22519;
            let b = rng.gen_biguint(255) % &p22519;
            let p = p22519.clone();

            additions.push(AddInstruction::new(a, b, p));
        }

        let stark = S {
            _marker: PhantomData,
        };

        let trace = stark.generate_trace(additions);

        // Verify proof as a stark
        let proof =
            prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default()).unwrap();
        verify_stark_proof(stark, proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof =
            add_virtual_stark_proof_with_pis(&mut recursive_builder, stark, &config, degree_bits);

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
    } */
    /*
    #[derive(Clone, Copy, Debug)]
    pub struct MulModLayoutCircuit;

    use crate::arithmetic::modular::mul;
    impl<F: RichField + Extendable<D>, const D: usize> EmulatedCircuitLayout<F, D, 1>
        for MulModLayoutCircuit
    {
        const PUBLIC_INPUTS: usize = 0;
        const NUM_ARITHMETIC_COLUMNS: usize = mul::NUM_ARITH_COLUMNS;
        const ENTRY_COLUMN: usize = 0;
        const TABLE_INDEX: usize = mul::NUM_ARITH_COLUMNS;

        type Layouts = ArithmeticLayout;
        const OPERATIONS: [ArithmeticLayout; 1] = [ArithmeticLayout::Mul(mul::MulModLayout::new(
            Register::Local(0, mul::N_LIMBS),
            Register::Local(mul::N_LIMBS, mul::N_LIMBS),
            Register::Local(2 * mul::N_LIMBS, mul::N_LIMBS),
            Register::Local(3 * mul::N_LIMBS, mul::NUM_OUTPUT_COLUMNS),
            Register::Local(
                4 * mul::N_LIMBS,
                mul::NUM_CARRY_COLUMNS + mul::NUM_WITNESS_COLUMNS,
            ),
        ))];
    }

    #[derive(Debug, Clone)]
    pub struct MulInstruction {
        pub a: BigUint,
        pub b: BigUint,
        pub modulus: BigUint,
    }

    impl MulInstruction {
        pub fn new(a: BigUint, b: BigUint, modulus: BigUint) -> Self {
            Self { a, b, modulus }
        }
    }

    impl<F: RichField + Extendable<D>, const D: usize> InstructionT<MulModLayoutCircuit, F, D, 1>
        for MulInstruction
    {
        fn generate_trace(self, pc: usize, tx: mpsc::Sender<(usize, usize, Vec<F>)>) {
            rayon::spawn(move || {
                let operation = ArithmeticOp::MulMod(self.a, self.b, self.modulus);
                let (trace_row, _) = operation.generate_trace_row();
                tx.send((pc, 0, trace_row)).unwrap();
            });
        }
    }

    #[test]
    fn test_arithmetic_stark_mul() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = CompliedStark<MulModLayoutCircuit, 1, F, D>;

        let num_rows = 2u64.pow(16);
        let config = StarkConfig::standard_fast_config();

        let p22519 = BigUint::from(2u32).pow(255) - BigUint::from(19u32);

        let mut rng = rand::thread_rng();

        let mut multiplication = Vec::new();

        for _ in 0..num_rows {
            let a: BigUint = rng.gen_biguint(255) % &p22519;
            let b = rng.gen_biguint(255) % &p22519;
            let p = p22519.clone();

            multiplication.push(MulInstruction::new(a, b, p));
        }

        let stark = S {
            _marker: PhantomData,
        };

        let trace = stark.generate_trace(multiplication);

        // Verify proof as a stark
        let proof =
            prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default()).unwrap();
        verify_stark_proof(stark, proof.clone(), &config).unwrap();

        // Verify recursive proof in a circuit
        let config_rec = CircuitConfig::standard_recursion_config();
        let mut recursive_builder = CircuitBuilder::<F, D>::new(config_rec);

        let degree_bits = proof.proof.recover_degree_bits(&config);
        let virtual_proof =
            add_virtual_stark_proof_with_pis(&mut recursive_builder, stark, &config, degree_bits);

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
    } */
}
