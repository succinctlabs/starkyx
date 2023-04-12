use anyhow::Result;

use super::den::Den;
use super::*;
use crate::arithmetic::builder::ChipBuilder;
use crate::arithmetic::chip::ChipParameters;
use crate::arithmetic::field::mul::{FpMul, FpMulConst};
use crate::arithmetic::field::quad::FpQuad;
use crate::arithmetic::trace::TraceHandle;

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct EcAddData<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize> {
    P: AffinePointRegister<E, N_LIMBS>,
    Q: AffinePointRegister<E, N_LIMBS>,
    R: AffinePointRegister<E, N_LIMBS>,
    XNUM: FpQuad<E::FieldParam, N_LIMBS>,
    YNUM: FpQuad<E::FieldParam, N_LIMBS>,
    PXPY: FpMul<E::FieldParam, N_LIMBS>,
    QXQY: FpMul<E::FieldParam, N_LIMBS>,
    PXPYQXQY: FpMul<E::FieldParam, N_LIMBS>,
    DXY: FpMulConst<E::FieldParam, N_LIMBS>,
    XDEN: Den<E::FieldParam, N_LIMBS>,
    YDEN: Den<E::FieldParam, N_LIMBS>,
}

impl<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize> EcAddData<E, N_LIMBS> {
    pub const fn num_ed_add_witness_columns() -> usize {
        2 * (FpQuad::<E::FieldParam, N_LIMBS>::num_quad_columns() - 4 * N_LIMBS)
            + 3 * (FpMul::<E::FieldParam, N_LIMBS>::num_mul_columns() - 2 * N_LIMBS)
            + 2 * (FpMulConst::<E::FieldParam, N_LIMBS>::num_mul_const_columns() - N_LIMBS)
            + 2 * (Den::<E::FieldParam, N_LIMBS>::num_den_columns() - 3 * N_LIMBS)
    }

    pub const fn num_ed_add_columns() -> usize {
        6 * N_LIMBS + Self::num_ed_add_witness_columns()
    }

    pub const fn num_ed_double_columns() -> usize {
        Self::num_ed_add_columns() - 3 * N_LIMBS
    }
}

pub trait FromEdwardsAdd<E: EdwardsParameters<N>, const N: usize>:
    From<FpMul<E::FieldParam, N>>
    + From<FpQuad<E::FieldParam, N>>
    + From<FpMulConst<E::FieldParam, N>>
    + From<Den<E::FieldParam, N>>
{
}

impl<L: ChipParameters<F, D>, F: RichField + Extendable<D>, const D: usize> ChipBuilder<L, F, D> {
    #[allow(non_snake_case)]
    pub fn ed_add<E: EdwardsParameters<N>, const N: usize>(
        &mut self,
        P: &AffinePointRegister<E, N>,
        Q: &AffinePointRegister<E, N>,
        result: &AffinePointRegister<E, N>,
    ) -> Result<EcAddData<E, N>>
    where
        L::Instruction: FromEdwardsAdd<E, N>,
    {
        let x_num_result = self.alloc_local::<FieldRegister<E::FieldParam, N>>()?;
        let y_num_result = self.alloc_local::<FieldRegister<E::FieldParam, N>>()?;
        let px_py_result = self.alloc_local::<FieldRegister<E::FieldParam, N>>()?;
        let qx_qy_result = self.alloc_local::<FieldRegister<E::FieldParam, N>>()?;
        let all_xy_result = self.alloc_local::<FieldRegister<E::FieldParam, N>>()?;
        let dxy_result = self.alloc_local::<FieldRegister<E::FieldParam, N>>()?;

        let x_num_ins = self.fpquad(&P.x, &Q.y, &Q.x, &P.y, &x_num_result)?;
        let y_num_ins = self.fpquad(&P.y, &Q.y, &P.x, &Q.x, &y_num_result)?;

        let px_py_ins = self.fpmul(&P.x, &P.y, &px_py_result)?;
        let qx_qy_ins = self.fpmul(&Q.x, &Q.y, &qx_qy_result)?;

        let all_xy_ins = self.fpmul(&px_py_result, &qx_qy_result, &all_xy_result)?;
        let dxy_ins = self.fpmul_const(&all_xy_result, E::D, &dxy_result)?;

        let r_x_ins = self.ed_den(&x_num_result, &dxy_result, true, &result.x)?;
        let r_y_ins = self.ed_den(&y_num_result, &dxy_result, false, &result.y)?;

        Ok(EcAddData {
            P: *P,
            Q: *Q,
            R: *result,
            XNUM: x_num_ins,
            YNUM: y_num_ins,
            PXPY: px_py_ins,
            QXQY: qx_qy_ins,
            PXPYQXQY: all_xy_ins,
            DXY: dxy_ins,
            XDEN: r_x_ins,
            YDEN: r_y_ins,
        })
    }

    #[allow(non_snake_case)]
    pub fn ed_double<E: EdwardsParameters<N>, const N: usize>(
        &mut self,
        P: &AffinePointRegister<E, N>,
        result: &AffinePointRegister<E, N>,
    ) -> Result<EcAddData<E, N>>
    where
        L::Instruction: FromEdwardsAdd<E, N>,
    {
        self.ed_add(P, P, result)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> TraceHandle<F, D> {
    #[allow(non_snake_case)]
    pub fn write_ed_add<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize>(
        &self,
        row_index: usize,
        P: &AffinePoint<E, N_LIMBS>,
        Q: &AffinePoint<E, N_LIMBS>,
        chip_data: EcAddData<E, N_LIMBS>,
    ) -> Result<AffinePoint<E, N_LIMBS>> {
        let x_num = self.write_fpquad(row_index, &P.x, &Q.y, &Q.x, &P.y, chip_data.XNUM)?;
        let y_num = self.write_fpquad(row_index, &P.y, &Q.y, &P.x, &Q.x, chip_data.YNUM)?;

        let px_py = self.write_fpmul(row_index, &P.x, &P.y, chip_data.PXPY)?;
        let qx_qy = self.write_fpmul(row_index, &Q.x, &Q.y, chip_data.QXQY)?;

        let all_xy = self.write_fpmul(row_index, &px_py, &qx_qy, chip_data.PXPYQXQY)?;
        let dxy = self.write_fpmul_const(row_index, &all_xy, chip_data.DXY)?;

        let r_x = self.write_ed_den(row_index, &x_num, &dxy, true, chip_data.XDEN)?;
        let r_y = self.write_ed_den(row_index, &y_num, &dxy, false, chip_data.YDEN)?;

        Ok(AffinePoint::new(r_x, r_y))
    }

    #[allow(non_snake_case)]
    pub fn write_ed_double<E: EdwardsParameters<N_LIMBS>, const N_LIMBS: usize>(
        &self,
        row_index: usize,
        P: &AffinePoint<E, N_LIMBS>,
        chip_data: EcAddData<E, N_LIMBS>,
    ) -> Result<AffinePoint<E, N_LIMBS>> {
        self.write_ed_add(row_index, P, P, chip_data)
    }
}

#[cfg(test)]
mod tests {

    use num::bigint::RandBigInt;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::timed;
    use plonky2::util::timing::TimingTree;
    use plonky2_maybe_rayon::*;
    use rand::thread_rng;

    use super::*;
    use crate::arithmetic::builder::ChipBuilder;
    use crate::arithmetic::chip::{ChipParameters, TestStark};
    use crate::arithmetic::ec::edwards::instructions::EdWardsMicroInstruction;
    use crate::arithmetic::trace::trace;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::verifier::verify_stark_proof;

    #[derive(Clone, Debug, Copy)]
    pub struct EdAddTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for EdAddTest {
        const NUM_ARITHMETIC_COLUMNS: usize =
            EcAddData::<Ed25519Parameters, 16>::num_ed_add_columns();
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = EdWardsMicroInstruction<Ed25519Parameters, 16>;
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_ed_add_row() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519Parameters;
        type S = TestStark<EdAddTest, F, D>;

        let _ = env_logger::builder().is_test(true).try_init();
        // build the stark
        let mut builder = ChipBuilder::<EdAddTest, F, D>::new();

        let P = builder.alloc_local_ec_point::<E, 16>().unwrap();
        let Q = builder.alloc_local_ec_point::<E, 16>().unwrap();
        let R = builder.alloc_local_ec_point::<E, 16>().unwrap();

        let ed_data = builder.ed_add::<E, 16>(&P, &Q, &R).unwrap();
        builder.write_ec_point(&P).unwrap();
        builder.write_ec_point(&Q).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let base = E::generator();
        let mut timing = TimingTree::new("Ed_Add row", log::Level::Debug);

        let trace = timed!(timing, "generate trace", {
            for i in 0..256usize {
                let handle = handle.clone();
                let base = base.clone();
                rayon::spawn(move || {
                    let mut rng = thread_rng();
                    let a = rng.gen_biguint(256);
                    let b = rng.gen_biguint(256);
                    let P_int = &base * &a;
                    let Q_int = &base * &b;
                    let R_exp = &P_int + &Q_int;

                    for j in 0..256usize {
                        handle.write_ec_point(256 * i + j, &P_int, &P).unwrap();
                        handle.write_ec_point(256 * i + j, &Q_int, &Q).unwrap();
                        let R = handle
                            .write_ed_add(256 * i + j, &P_int, &Q_int, ed_data)
                            .unwrap();
                        assert_eq!(R, R_exp);
                    }
                });
            }
            drop(handle);

            generator.generate_trace(&chip, num_rows as usize).unwrap()
        });

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

    #[derive(Clone, Debug, Copy)]
    pub struct EdDoubleTest;

    impl<F: RichField + Extendable<D>, const D: usize> ChipParameters<F, D> for EdDoubleTest {
        const NUM_ARITHMETIC_COLUMNS: usize =
            EcAddData::<Ed25519Parameters, 16>::num_ed_double_columns();
        const NUM_FREE_COLUMNS: usize = 0;

        type Instruction = EdWardsMicroInstruction<Ed25519Parameters, 16>;
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_ed_double_row() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type E = Ed25519Parameters;
        type S = TestStark<EdAddTest, F, D>;

        let _ = env_logger::builder().is_test(true).try_init();
        // build the stark
        let mut builder = ChipBuilder::<EdAddTest, F, D>::new();

        let P = builder.alloc_ec_point::<E, 16>().unwrap();
        let R = builder.alloc_ec_point::<E, 16>().unwrap();

        let ed_double_data = builder.ed_double::<E, 16>(&P, &R).unwrap();
        builder.write_ec_point(&P).unwrap();

        let (chip, spec) = builder.build();

        // Construct the trace
        // Construct the trace
        let num_rows = 2u64.pow(16);
        let (handle, generator) = trace::<F, D>(spec);

        let base = E::generator();
        let mut timing = TimingTree::new("Ed_double row", log::Level::Debug);

        let trace = timed!(timing, "generate trace", {
            for i in 0..256usize {
                let handle = handle.clone();
                let base = base.clone();
                rayon::spawn(move || {
                    let mut rng = thread_rng();
                    let a = rng.gen_biguint(256);
                    let P_int = &base * &a;
                    let R_exp = &P_int + &P_int;

                    for j in 0..256usize {
                        handle.write_ec_point(256 * i + j, &P_int, &P).unwrap();
                        let R = handle
                            .write_ed_double(256 * i + j, &P_int, ed_double_data)
                            .unwrap();
                        assert_eq!(R, R_exp);
                    }
                });
            }
            drop(handle);

            generator.generate_trace(&chip, num_rows as usize).unwrap()
        });

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
