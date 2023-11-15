//! A STARK to compute several Fibonacci functions in a single stark.

use curta::chip::register::element::ElementRegister;
use curta::math::goldilocks::cubic::GoldilocksCubicParameters;
use curta::plonky2::stark::config::CurtaPoseidonGoldilocksConfig;
use curta::prelude::*;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::util::timing::TimingTree;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciSIMD;

impl AirParameters for FibonacciSIMD {
    type Field = GoldilocksField;
    type CubicParams = GoldilocksCubicParameters;

    type Instruction = EmptyInstruction<GoldilocksField>;

    const NUM_ARITHMETIC_COLUMNS: usize = 0;
    const NUM_FREE_COLUMNS: usize = 11;
    const EXTENDED_COLUMNS: usize = 21;
}

fn fibonacci(n: usize) -> u64 {
    (0..n).fold((0, 1), |(x_0, x_1), _| (x_1, x_0 + x_1)).1
}

fn main() {
    type L = FibonacciSIMD;
    type F = GoldilocksField;
    type C = CurtaPoseidonGoldilocksConfig;

    env_logger::init();

    let mut builder = StarkBuilder::<L>::new();

    let fib_len = 32;
    let num_ops = 1024;

    // Initialze auxilary variable for SIMD control flow. These are explicit to allow the user to
    // have maximum control, but this comes at a cost of some bioler-plate code.

    // Make a cycle of length 32 which gives us access to a start bit and an end bit. We will only
    // be using the end bit in this example.
    let cycle_fib = builder.cycle(5);
    // Using the end bit, we can get a process_id, which will give us a unique identifier for each
    // computation thread in the AIR.
    let idx = builder.process_id(fib_len, cycle_fib.end_bit);
    let end_flag = builder.expression::<ElementRegister>(cycle_fib.end_bit.not_expr());

    // Initialize slice pointers for the values of x_0 and x_1.
    let x_0_slice = builder.uninit_slice();
    let x_1_slice = builder.uninit_slice();

    let x_0_init = builder.constant::<ElementRegister>(&F::ZERO);
    let x_1_init = builder.constant::<ElementRegister>(&F::ONE);
    let x_1_final = builder.constant::<ElementRegister>(&F::from_canonical_u64(fibonacci(fib_len)));

    for i in 0..num_ops {
        let index = i * fib_len;
        builder.store(&x_0_slice.get(i), x_0_init, &Time::constant(index), None);
        builder.store(&x_1_slice.get(i), x_1_init, &Time::constant(index), None);

        builder.free(
            &x_1_slice.get(i),
            x_1_final,
            &Time::constant(index + fib_len),
        );
    }

    let clk = Time::from_element(builder.clk());
    let x_0 = builder.load::<ElementRegister>(&x_0_slice.get_at(idx), &clk);
    let x_1 = builder.load::<ElementRegister>(&x_1_slice.get_at(idx), &clk);

    let sum = builder.add(x_0, x_1);
    builder.store(&x_1_slice.get_at(idx), sum, &clk.advance(), None);
    builder.store(&x_0_slice.get_at(idx), x_1, &clk.advance(), Some(end_flag));

    let num_rows = num_ops * fib_len;
    let stark = builder.build::<C, 2>(num_rows);

    let mut writer_data = AirWriterData::new(&stark.air_data, num_rows);

    // Write the initial values to the stark.
    let mut public_writer = writer_data.public_writer();
    let air_data = &stark.air_data;
    air_data.write_global_instructions(&mut public_writer);
    writer_data.chunks_par(fib_len).for_each(|mut chunk| {
        for i in 0..fib_len {
            let mut writer = chunk.window_writer(i);
            air_data.write_trace_instructions(&mut writer);
        }
    });

    let (trace, public) = (writer_data.trace, writer_data.public);

    let proof = stark
        .prove(&trace, &public, &mut TimingTree::default())
        .unwrap();

    stark.verify(proof.clone(), &public).unwrap();
}
