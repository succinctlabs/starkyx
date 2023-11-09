// pub mod keccak256;
use crate::chip::register::array::ArrayRegister;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;

pub struct Keccak256;

pub trait KeccakAir<L: AirParameters>
where
    L::Instruction: UintInstructions,
{
    const RHO_OFFSETS: [[usize; 5]; 5];

    fn keccak_p(
        builder: &mut BytesBuilder<L>,
        state: ArrayRegister<U64Register>,
    ) -> ArrayRegister<U64Register>;
}

impl<L: AirParameters> KeccakAir<L> for Keccak256
where
    L::Instruction: UintInstructions,
{
    const RHO_OFFSETS: [[usize; 5]; 5] = [
        [0, 1, 190, 28, 91],
        [36, 300, 6, 55, 276],
        [3, 10, 171, 153, 231],
        [105, 45, 15, 21, 136],
        [210, 66, 253, 120, 78],
    ];

    fn keccak_p(
        builder: &mut BytesBuilder<L>,
        state: ArrayRegister<U64Register>,
    ) -> ArrayRegister<U64Register> {
        let mut c: Vec<U64Register> = Vec::new();
        let mut d: Vec<U64Register> = Vec::new();

        // C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4], for x in 0…4
        for x in 0..5 {
            let mut c_temp = builder.xor(&state.get(x), &state.get(x + 5));
            c_temp = builder.xor(&c_temp, &state.get(x + 10));
            c_temp = builder.xor(&c_temp, &state.get(x + 15));
            c.push(builder.xor(&c_temp, &state.get(x + 20)));
        }

        // D[x] = C[x-1] xor rot(C[x+1],1), for x in 0…4
        for x in 0..5 {
            let d_temp = builder.xor(&c[(x + 4) % 5], &c[(x + 1) % 5]);
            d.push(builder.rotate_left(d_temp, 1));
        }

        // A[x,y] = A[x,y] xor D[x], for (x,y) in (0…4,0…4)
        let mut state_temp: Vec<U64Register> = Vec::new();

        for y in 0..5 {
            for x in 0..5 {
                state_temp.push(builder.xor(&state.get(x + y * 5), &d[x]));
            }
        }

        // ############################################
        // Rho
        // ############################################

        let mut rho_x = 0;
        let mut rho_y = 1;

        const RHO_OFFSETS: [[usize; 5]; 5] = [
            [0, 1, 190, 28, 91],
            [36, 300, 6, 55, 276],
            [3, 10, 171, 153, 231],
            [105, 45, 15, 21, 136],
            [210, 66, 253, 120, 78],
        ];

        for _ in 0..24 {
            // Rotate each lane by an offset
            let index = rho_x + 5 * rho_y;
            state_temp[index] =
                builder.rotate_left(state_temp[index], RHO_OFFSETS[rho_y][rho_x] % 64);
            let rho_x_prev = rho_x;
            rho_x = rho_y;
            rho_y = (2 * rho_x_prev + 3 * rho_y) % 5;
        }

        // ############################################
        // Pi
        // ############################################

        state
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::util::timing::TimingTree;
    use serde::{Deserialize, Serialize};

    use crate::chip::arithmetic::expression::ArithmeticExpression;
    use crate::chip::builder::tests::GoldilocksCubicParameters;
    use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::uint::register::U64Register;
    use crate::chip::AirParameters;
    use crate::machine::bytes::builder::BytesBuilder;
    use crate::machine::hash::keccak::{Keccak256, KeccakAir};
    use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Keccak256Test;

    impl AirParameters for Keccak256Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 1729;
        const EXTENDED_COLUMNS: usize = 954;
    }

    #[test]
    fn test_keccak_p() {
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_byte_multi_stark", log::Level::Debug);

        let mut builder = BytesBuilder::<Keccak256Test>::new();

        let num_digests = 25;
        let state = builder.api.alloc_array_public::<U64Register>(num_digests);

        for i in 0..num_digests {
            builder.api.set_to_expression_public(
                &state.get(i),
                ArithmeticExpression::from_constant_vec(vec![GoldilocksField::ZERO; 8]),
            );
        }

        Keccak256::keccak_p(&mut builder, state);

        let num_rows = 1 << 6;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);
        // NOTE: you always need to write something to the trace even if you have zero values, otherwise the lookup argument will fail at the proof level
        writer.write_global_instructions(&stark.air_data);
        for i in 0..num_rows {
            writer.write_row_instructions(&stark.air_data, i);
        }

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();
    }
}
