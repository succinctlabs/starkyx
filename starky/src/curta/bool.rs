use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::air::parser::AirParser;
use super::builder::StarkBuilder;
use super::chip::StarkParameters;
use super::instruction::Instruction;
use super::register::{BitRegister, MemorySlice, Register};
use super::trace::writer::TraceWriter;

#[derive(Debug, Clone, Copy)]
pub struct SelectInstruction<T> {
    bit: BitRegister,
    true_value: T,
    false_value: T,
    pub result: T,
}

impl<L: StarkParameters<F, D>, F: RichField + Extendable<D>, const D: usize> StarkBuilder<L, F, D> {
    pub fn select<T: Register>(&mut self, bit: &BitRegister, a: &T, b: &T) -> SelectInstruction<T>
    where
        L::Instruction: From<SelectInstruction<T>>,
    {
        let result = self.alloc::<T>();
        let instr = SelectInstruction {
            bit: *bit,
            true_value: *a,
            false_value: *b,
            result,
        };
        self.constrain_instruction(instr.into()).unwrap();
        instr
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceWriter<F, D> {
    #[allow(dead_code)]
    pub fn write_bit(&self, row_index: usize, bit: bool, data: &BitRegister) -> Result<()> {
        self.write_data(row_index, *data, vec![F::from_canonical_u16(bit as u16)])
    }
}

impl<F: RichField + Extendable<D>, const D: usize, T: Register> Instruction<F, D>
    for SelectInstruction<T>
{
    fn trace_layout(&self) -> Vec<MemorySlice> {
        vec![*self.result.register()]
    }

    fn eval<AP: AirParser<Field = F>>(&self, parser: &mut AP) -> Vec<AP::Var> {
        let bit = self.bit.eval(parser);
        let true_slice = self.true_value.register().eval_slice(parser).to_vec();
        let false_slice = self.false_value.register().eval_slice(parser).to_vec();
        let result_slice = self.result.register().eval_slice(parser).to_vec();

        let one = parser.one();
        let one_minus_bit = parser.sub(one, bit);

        true_slice
            .iter()
            .zip(false_slice.iter())
            .zip(result_slice.iter())
            .map(|((x_true, x_false), x)| {
                let bit_x_true = parser.mul(*x_true, bit);
                let one_minus_bit_x_false = parser.mul(*x_false, one_minus_bit);
                let expected_res = parser.add(bit_x_true, one_minus_bit_x_false);
                parser.sub(expected_res, *x)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::extension::Extendable;
    use plonky2::field::types::Field;
    use plonky2::hash::hash_types::RichField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::config::StarkConfig;
    use crate::curta::builder::StarkBuilder;
    use crate::curta::chip::{ChipStark, StarkParameters};
    use crate::curta::extension::cubic::goldilocks_cubic::GoldilocksCubicParameters;
    use crate::curta::register::BitRegister;
    use crate::curta::stark::prover::prove;
    use crate::curta::stark::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::curta::stark::verifier::verify_stark_proof;
    use crate::curta::trace::arithmetic::{trace, ArithmeticGenerator};

    #[derive(Debug, Clone, Copy)]
    pub struct BoolTest;

    impl<F: RichField + Extendable<D>, const D: usize> StarkParameters<F, D> for BoolTest {
        const NUM_ARITHMETIC_COLUMNS: usize = 10;
        const NUM_FREE_COLUMNS: usize = 27;
        type Instruction = SelectInstruction<BitRegister>;
    }

    #[test]
    fn test_bool_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = GoldilocksCubicParameters;
        type S = ChipStark<BoolTest, F, D>;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("stark_proof", log::Level::Debug);

        let mut builder = StarkBuilder::<BoolTest, F, D>::new();

        let bit_one = builder.alloc::<BitRegister>();
        builder.write_data(&bit_one).unwrap();
        let bit_zero = builder.alloc::<BitRegister>();
        builder.write_data(&bit_zero).unwrap();

        let dummy = builder.alloc::<BitRegister>();
        builder.write_data(&dummy).unwrap();

        let chip = builder.build();

        // Test successful proof
        // Construct the trace
        let num_rows = 2u64.pow(5) as usize;
        let (handle, generator) = trace::<F, E, D>(num_rows);
        for i in 0..num_rows {
            handle.write_data(i, bit_one, vec![F::ONE]).unwrap();
            handle.write_data(i, bit_zero, vec![F::ZERO]).unwrap();
            handle.write_data(i, dummy, vec![F::ZERO]).unwrap();
        }
        drop(handle);

        // Generate the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip.clone());
        let proof = prove::<F, C, S, ArithmeticGenerator<F, E, D>, D, 2>(
            stark.clone(),
            &config,
            generator,
            num_rows,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Generate the recursive proof.
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
        verify_stark_proof_circuit::<F, C, S, D, 2>(
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
        timed!(
            timing,
            "verify recursive proof",
            recursive_data.verify(recursive_proof).unwrap()
        );
        timing.print();

        // test unsuccesfull proof
        // Construct the trace
        let num_rows = 2u64.pow(5) as usize;
        let (handle, generator) = trace::<F, E, D>(num_rows);
        for i in 0..num_rows {
            handle.write_data(i, bit_zero, vec![F::ZERO]).unwrap();
            handle
                .write_data(i, bit_one, vec![F::ONE + F::ONE])
                .unwrap();
            handle.write_data(i, dummy, vec![F::ZERO; 1]).unwrap();
        }
        drop(handle);

        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip);

        // Verify proof as a stark
        let proof = prove::<F, C, S, ArithmeticGenerator<F, E, D>, D, 2>(
            stark.clone(),
            &config,
            generator,
            num_rows,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();

        let res = verify_stark_proof(stark.clone(), proof.clone(), &config);
        assert!(res.is_err())
    }

    #[test]
    fn test_selector() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = GoldilocksCubicParameters;
        type S = ChipStark<BoolTest, F, D>;

        let _ = env_logger::builder().is_test(true).try_init();
        let mut timing = TimingTree::new("stark_proof", log::Level::Debug);

        let mut builder = StarkBuilder::<BoolTest, F, D>::new();
        let bit = builder.alloc::<BitRegister>();
        builder.write_data(&bit).unwrap();
        let x = builder.alloc::<BitRegister>();
        builder.write_data(&x).unwrap();
        let y = builder.alloc::<BitRegister>();
        builder.write_data(&y).unwrap();
        let sel = builder.select(&bit, &x, &y);
        let chip = builder.build();

        // Test successful proof
        // Construct the trace
        let num_rows = 2u64.pow(5) as usize;
        let (handle, generator) = trace::<F, E, D>(num_rows);

        for i in 0..num_rows {
            let x_i = 0u16;
            let y_i = 1u16;
            let bit_i = if i % 2 == 0 { true } else { false };
            handle.write_bit(i, bit_i, &bit).unwrap();
            let res = if i % 2 == 0 { x_i } else { y_i };
            handle
                .write_data(i, x, vec![F::from_canonical_u16(x_i)])
                .unwrap();
            handle
                .write_data(i, y, vec![F::from_canonical_u16(y_i)])
                .unwrap();
            handle
                .write(i, sel, vec![F::from_canonical_u16(res)])
                .unwrap();
        }
        drop(handle);

        // Generate the proof.
        let config = StarkConfig::standard_fast_config();
        let stark = ChipStark::new(chip);
        let proof = prove::<F, C, S, ArithmeticGenerator<F, E, D>, D, 2>(
            stark.clone(),
            &config,
            generator,
            num_rows,
            [],
            &mut TimingTree::default(),
        )
        .unwrap();
        verify_stark_proof(stark.clone(), proof.clone(), &config).unwrap();

        // Generate the recursive proof.
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
        verify_stark_proof_circuit::<F, C, S, D, 2>(
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
        timed!(
            timing,
            "verify recursive proof",
            recursive_data.verify(recursive_proof).unwrap()
        );
        timing.print();
    }
}
