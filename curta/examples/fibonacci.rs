//! A Fibonacci STARK.

use curta::chip::register::element::ElementRegister;
use curta::math::goldilocks::cubic::GoldilocksCubicParameters;
use curta::plonky2::stark::config::CurtaPoseidonGoldilocksConfig;
use curta::prelude::*;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::util::timing::TimingTree;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fibonacci;

impl AirParameters for Fibonacci {
    type Field = GoldilocksField;
    type CubicParams = GoldilocksCubicParameters;

    type Instruction = EmptyInstruction<GoldilocksField>;

    const NUM_ARITHMETIC_COLUMNS: usize = 0;
    const NUM_FREE_COLUMNS: usize = 3;
    const EXTENDED_COLUMNS: usize = 3;
}

fn main() {
    type L = Fibonacci;
    type F = GoldilocksField;
    type C = CurtaPoseidonGoldilocksConfig;

    env_logger::init();

    let mut builder = StarkBuilder::<L>::new();

    let x_0 = builder.alloc::<ElementRegister>();
    let x_1 = builder.alloc::<ElementRegister>();

    builder.set_first_row_const(&x_0, &F::ZERO);
    builder.set_first_row_const(&x_1, &F::ONE);

    builder.set_next(&x_0, &x_1);
    builder.set_next_expression(&x_1, x_0.expr() + x_1.expr());

    let num_rows = 1 << 5;
    let stark = builder.build::<C, 2>(num_rows);

    let mut writer_data = AirWriterData::new(&stark.air_data, num_rows);

    let air_data = &stark.air_data;
    air_data.write_global_instructions(&mut writer_data.public_writer());
    writer_data.chunks(num_rows).for_each(|mut chunk| {
        for i in 0..num_rows {
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
