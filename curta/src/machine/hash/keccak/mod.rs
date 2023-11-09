use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::machine::hash::sha::sha256::ROUND_CONSTANTS;
use crate::math::prelude::*;
pub struct Keccak256;

pub trait KeccakAir<L: AirParameters>
where
    L::Instruction: UintInstructions,
{
    const RHO_OFFSETS: [[usize; 5]; 5];

    const NUM_WORDS: usize;

    // fn to_words(bytes: &[u8]) -> Vec<u64>;

    // fn pad(n_bytes: usize) -> Vec<u8>;

    fn keccak_p(
        builder: &mut BytesBuilder<L>,
        state: ArrayRegister<U64Register>,
    ) -> ArrayRegister<U64Register>;

    fn hash(builder: &mut BytesBuilder<L>, input: &[u8]) -> [u8; 32];
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

    const NUM_WORDS: usize = 25;

    fn keccak_p(
        builder: &mut BytesBuilder<L>,
        state: ArrayRegister<U64Register>,
    ) -> ArrayRegister<U64Register> {
        const RC: [u64; 24] = [
            1u64,
            0x8082u64,
            0x800000000000808au64,
            0x8000000080008000u64,
            0x808bu64,
            0x80000001u64,
            0x8000000080008081u64,
            0x8000000000008009u64,
            0x8au64,
            0x88u64,
            0x80008009u64,
            0x8000000au64,
            0x8000808bu64,
            0x800000000000008bu64,
            0x8000000000008089u64,
            0x8000000000008003u64,
            0x8000000000008002u64,
            0x8000000000000080u64,
            0x800au64,
            0x800000008000000au64,
            0x8000000080008081u64,
            0x8000000000008080u64,
            0x80000001u64,
            0x8000000080008008u64,
        ];

        // Initialize the round constants and set them to the constant value.
        let round_constant_values =
            builder.constant_array::<U64Register>(&RC.map(u64_to_le_field_bytes));

        let num_words = 25;
        let mut state_temp: Vec<U64Register> = Vec::with_capacity(25);

        for round in 0..24 {
            let mut c: Vec<U64Register> = Vec::new();
            let mut d: Vec<U64Register> = Vec::new();
            let mut b: Vec<U64Register> = Vec::new();

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
                d.push(builder.rotate_right(d_temp, 1));
            }

            // A[x,y] = A[x,y] xor D[x], for (x,y) in (0…4,0…4)
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

            // TODO: save that to the trace
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
                    builder.rotate_right(state_temp[index], RHO_OFFSETS[rho_y][rho_x] % 64);
                let rho_x_prev = rho_x;
                rho_x = rho_y;
                rho_y = (2 * rho_x_prev + 3 * rho_y) % 5;
            }

            // ############################################
            // Pi
            // ############################################

            // B[y,2*x+3*y] = rot(A[x,y], r[x,y]), for (x,y) in (0…4,0…4)
            for y in 0..5 {
                for x in 0..5 {
                    let index = ((x + 3 * y) % 5) + x * 5;
                    let temp = builder.alloc::<U64Register>();
                    builder.set_to_expression(&temp, state_temp[index].expr());
                    b.push(temp);
                }
            }

            // ############################################
            // Chi
            // ############################################

            // A[x,y] = B[x,y] xor ((not B[x+1,y]) and B[x+2,y]), for (x,y) in (0…4,0…4)
            for y in 0..5 {
                for x in 0..5 {
                    let index = x + y * 5;

                    let mut temp = builder.not(&b[(x + 1) % 5 + y * 5]);
                    temp = builder.and(&temp, &b[(x + 2) % 5 + y * 5]);
                    state_temp[index] = builder.xor(&b[index], &temp);
                }
            }

            // ############################################
            // Iota
            // ############################################

            // A[0,0] = A[0,0] xor RC
            state_temp[0] = builder.xor(&state_temp[0], &round_constant_values.get(round));
        }

        let new_state = builder.alloc_array::<U64Register>(num_words);

        for i in 0..num_words {
            builder.set_to_expression(&new_state.get(i), state_temp[i].expr());
        }

        new_state
        // state
    }

    fn hash(builder: &mut BytesBuilder<L>, input: &[u8]) -> [u8; 32] {
        const OUTPUT_LEN: usize = 256;
        const CAPACITY: usize = OUTPUT_LEN * 2;
        const STATE_WIDTH: usize = 1600;
        const RATE: usize = STATE_WIDTH - CAPACITY;
        if input.len() > (RATE / 8) {
            panic!("Sponge not supported");
        };

        fn pad(n_bytes: usize) -> Vec<u8> {
            const OUTPUT_LEN: usize = 256;
            const CAPACITY: usize = OUTPUT_LEN * 2;
            const STATE_WIDTH: usize = 1600;
            const RATE: usize = STATE_WIDTH - CAPACITY;
            let padded_size_in_bytes = n_bytes.next_multiple_of(RATE / 8);
            let mut padding = vec![0u8; padded_size_in_bytes - n_bytes];
            let padding_len = padding.len();
            padding[0] = 0x01;
            padding[padding_len - 1] = 0x80;
            padding
        }

        fn to_words(bytes: &[u8]) -> Vec<u64> {
            let mut bytes_resized = bytes.to_vec();
            bytes_resized.resize(bytes.len().next_multiple_of(8), 0);

            let mut words = vec![0u64; bytes_resized.len() / 8];
            for i in 0..words.len() {
                words[i] = u64::from_le_bytes([
                    bytes_resized[i * 8],
                    bytes_resized[i * 8 + 1],
                    bytes_resized[i * 8 + 2],
                    bytes_resized[i * 8 + 3],
                    bytes_resized[i * 8 + 4],
                    bytes_resized[i * 8 + 5],
                    bytes_resized[i * 8 + 6],
                    bytes_resized[i * 8 + 7],
                ]);
            }
            words
        }

        // Padding
        let pad_vec: Vec<u8> = pad(input.len());
        let mut p = input.to_vec();
        p.extend(pad_vec);

        // Transform padded input into array of words
        let mut words = to_words(&p);
        words.resize(25, 0);
        // let mut state: [u64; 25] = words.try_into().unwrap();
        let state = builder.api.alloc_array_public::<U64Register>(25);

        for i in 0..25 {
            builder.api.set_to_expression_public(
                &state.get(i),
                ArithmeticExpression::from_constant_vec(vec![L::Field::ZERO; 8]),
            );
        }

        // The permutation
        Self::keccak_p(builder, state);

        // Output in bytes (32 bytes)
        // let out = [state[0], state[1], state[2], state[3]];
        // let mut out_bytes = [0u8; 32];
        // out_bytes[0..8].copy_from_slice(&out[0].to_le_bytes());
        // out_bytes[8..16].copy_from_slice(&out[1].to_le_bytes());
        // out_bytes[16..24].copy_from_slice(&out[2].to_le_bytes());
        // out_bytes[24..32].copy_from_slice(&out[3].to_le_bytes());

        // out_bytes
        [0; 32]
    }
}

#[cfg(test)]
mod tests {
    use core::num;

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
        const NUM_FREE_COLUMNS: usize = 75841;
        const EXTENDED_COLUMNS: usize = 44754;
    }

    #[test]
    fn test_keccak_p() {
        type C = CurtaPoseidonGoldilocksConfig;
        type _Config = <C as CurtaConfig<2>>::GenericConfig;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_byte_multi_stark", log::Level::Debug);

        let mut builder = BytesBuilder::<Keccak256Test>::new();

        let num_words = 25;
        let state = builder.api.alloc_array_public::<U64Register>(num_words);

        for i in 0..num_words {
            builder.api.set_to_expression_public(
                &state.get(i),
                ArithmeticExpression::from_constant_vec(vec![GoldilocksField::ZERO; 8]),
            );
        }

        let new_state = Keccak256::keccak_p(&mut builder, state);

        let num_rows = 1 << 6;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);
        // NOTE: you always need to write something to the trace even if you have zero values, otherwise the lookup argument will fail at the proof level
        writer.write_global_instructions(&stark.air_data);
        for i in 0..num_rows {
            writer.write_row_instructions(&stark.air_data, i);
            println!("{:?}", writer.read_array::<U64Register, 25>(&new_state, i));
        }

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();
    }

    #[test]
    fn test_keccak256_hash() {}
}
