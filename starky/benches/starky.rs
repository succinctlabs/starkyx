//! Benchmarks Starky proof generation for a range of columns, rows, and constraint degrees.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use plonky2::field::types::Field;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use pprof::criterion::{Output, PProfProfiler};
use seq_macro::seq;
use starky::benchmark::BenchmarkStark;
use starky::config::StarkConfig;
use starky::prover::prove;

fn bench(c: &mut Criterion) {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let mut group = c.benchmark_group("starky-prover");
    group.sample_size(10);

    // Sensitivity Analysis: Columns
    {
        const SWEEP_NUM_COLS: [usize; 12] =
            [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
        const BASE_NUM_ROWS: usize = 65536;
        const BASE_CONSTRAINT_DEGREE_1: usize = 2;
        seq!(i in 0..12 {
            {
                const NUM_COLS: usize = SWEEP_NUM_COLS[i];
                type S = BenchmarkStark<F, D, NUM_COLS, BASE_NUM_ROWS, BASE_CONSTRAINT_DEGREE_1>;
                let config = StarkConfig::standard_fast_config();
                let public_inputs = [F::TWO];
                let stark = S::new();
                group.bench_function(BenchmarkId::new("sensitivity-cols-with-65536-rows-and-degree-2", NUM_COLS), |b| {
                    b.iter(|| {
                        let trace = stark.generate_trace();
                        prove::<F, C, S, D>(
                            stark,
                            &config,
                            trace,
                            public_inputs,
                            &mut TimingTree::default(),
                        )
                        .unwrap();
                    })
                });
            }
        });
    }

    // Sensitivity Analysis: Rows
    {
        const SWEEP_NUM_ROWS: [usize; 14] = [
            32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
        ];
        const BASE_NUM_COLS: usize = 32;
        const BASE_CONSTRAINT_DEGREE: usize = 2;
        seq!(i in 0..14 {
            {
                const NUM_ROWS: usize = SWEEP_NUM_ROWS[i];
                type S = BenchmarkStark<F, D, BASE_NUM_COLS, NUM_ROWS, BASE_CONSTRAINT_DEGREE>;
                let config = StarkConfig::standard_fast_config();
                let public_inputs = [F::TWO];
                let stark = S::new();
                group.bench_function(BenchmarkId::new("sensitivity-rows-with-32-cols-and-degree-2", NUM_ROWS), |b| {
                    b.iter(|| {
                        let trace = stark.generate_trace();
                        prove::<F, C, S, D>(
                            stark,
                            &config,
                            trace,
                            public_inputs,
                            &mut TimingTree::default(),
                        )
                        .unwrap();
                    })
                });
            }
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(10, Output::Flamegraph(None)));
    targets = bench
}
criterion_main!(benches);
