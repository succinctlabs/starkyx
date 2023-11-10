use itertools::Itertools;

use crate::chip::arithmetic::expression::ArithmeticExpression;
use crate::chip::memory::time::Time;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::Register;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::u64_to_le_field_bytes;
use crate::chip::AirParameters;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
use crate::math::prelude::*;

pub struct KeccakPure;

impl KeccakPure {
    // Keccak256 parameters in bits
    const ROUNDS: usize = 24;
    const OUTPUT_LEN: usize = 256;
    const CAPACITY: usize = Self::OUTPUT_LEN * 2;
    const STATE_WIDTH: usize = 1600;
    const RATE: usize = Self::STATE_WIDTH - Self::CAPACITY;

    // Table 2 of https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
    const RHO_OFFSETS: [[u32; 5]; 5] = [
        [0, 1, 190, 28, 91],
        [36, 300, 6, 55, 276],
        [3, 10, 171, 153, 231],
        [105, 45, 15, 21, 136],
        [210, 66, 253, 120, 78],
    ];

    // Copied from https://github.com/debris/tiny-keccak/blob/master/src/keccakf.rs
    const RC: [u64; Self::ROUNDS] = [
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

    // Multi-rate padding as described in Section 5 of https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
    fn pad(n_bytes: usize) -> Vec<u8> {
        let padded_size_in_bytes = n_bytes.next_multiple_of(Self::RATE / 8);
        let mut padding = vec![0u8; padded_size_in_bytes - n_bytes];
        let padding_len = padding.len();
        padding[0] = 0x01;
        padding[padding_len - 1] = 0x80;
        padding
    }

    pub fn keccak256(input: &[u8]) -> [u8; 32] {
        if input.len() > (Self::RATE / 8) {
            panic!("Sponge not supported");
        };

        // Padding
        let pad_vec = Self::pad(input.len());
        let mut p = input.to_vec();
        p.extend(pad_vec);

        // Transform padded input into array of words
        let mut words = Self::to_words(&p);
        words.resize(25, 0);
        let mut state: [u64; 25] = words.try_into().unwrap();

        // The permutation
        Self::keccak_p(&mut state);

        // Output in bytes (32 bytes)
        let out = [state[0], state[1], state[2], state[3]];
        let mut out_bytes = [0u8; 32];
        out_bytes[0..8].copy_from_slice(&out[0].to_le_bytes());
        out_bytes[8..16].copy_from_slice(&out[1].to_le_bytes());
        out_bytes[16..24].copy_from_slice(&out[2].to_le_bytes());
        out_bytes[24..32].copy_from_slice(&out[3].to_le_bytes());

        out_bytes
    }

    pub fn keccak_p(state: &mut [u64; 25]) {
        for i in 0..Self::ROUNDS {
            // ############################################
            // Theta
            // ############################################

            let mut c = [0; 5];
            let mut d = [0; 5];

            // C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4], for x in 0…4
            for y in 0..5 {
                for x in 0..5 {
                    c[x] ^= state[x + y * 5];
                }
            }

            // D[x] = C[x-1] xor rot(C[x+1],1), for x in 0…4
            for x in 0..5 {
                d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            }

            // A[x,y] = A[x,y] xor D[x], for (x,y) in (0…4,0…4)
            for y in 0..5 {
                for x in 0..5 {
                    state[x + y * 5] ^= d[x];
                }
            }

            // ############################################
            // Rho
            // ############################################

            let mut rho_x = 0;
            let mut rho_y = 1;

            for _ in 0..24 {
                // Rotate each lane by an offset
                let index = rho_x + 5 * rho_y;
                state[index] = state[index].rotate_left(Self::RHO_OFFSETS[rho_y][rho_x] % 64);
                let rho_x_prev = rho_x;
                rho_x = rho_y;
                rho_y = (2 * rho_x_prev + 3 * rho_y) % 5;
            }

            // ############################################
            // Pi
            // ############################################

            let state_cloned = state.clone();
            // B[y,2*x+3*y] = rot(A[x,y], r[x,y]), for (x,y) in (0…4,0…4)
            for y in 0..5 {
                for x in 0..5 {
                    let index = ((x + 3 * y) % 5) + x * 5;
                    state[x + y * 5] = state_cloned[index];
                }
            }

            // ############################################
            // Chi
            // ############################################

            let state_cloned = state.clone();
            // A[x,y] = B[x,y] xor ((not B[x+1,y]) and B[x+2,y]), for (x,y) in (0…4,0…4)
            for y in 0..5 {
                for x in 0..5 {
                    let index = x + y * 5;
                    state[index] = state_cloned[index]
                        ^ (!state_cloned[(x + 1) % 5 + y * 5]) & state_cloned[(x + 2) % 5 + y * 5];
                }
            }

            // ############################################
            // Iota
            // ############################################

            // A[0,0] = A[0,0] xor RC
            state[0] ^= Self::RC[i];
        }
    }
}
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
        state_input: ArrayRegister<U64Register>,
    ) -> ArrayRegister<U64Register> {
        const NUM_WORDS: usize = 25;

        const RHO_OFFSETS: [[usize; 5]; 5] = [
            [0, 1, 190, 28, 91],
            [36, 300, 6, 55, 276],
            [3, 10, 171, 153, 231],
            [105, 45, 15, 21, 136],
            [210, 66, 253, 120, 78],
        ];

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

        // const ROUNDS: usize = 24;

        // let round_end_bit = builder.alloc::<BitRegister>();
        // let clk = builder.clk;

        // `process_id` is a register is computed by counting the number of cycles. We do this by
        // setting `process_id` to be the cumulative sum of the `end_bit` of each cycle.
        // let process_id = builder.process_id(ROUNDS, round_end_bit);

        // Initialize the round constants and set them to the constant value.
        let round_constant_values =
            builder.constant_array::<U64Register>(&RC.map(u64_to_le_field_bytes));

        // let round_constants_reg = builder.alloc::<U64Register>();

        // builder.set_to_expression_transition(&round_constants_reg, clk *);

        let mut state: Vec<U64Register> = (0..NUM_WORDS).map(|i| state_input.get(i)).collect();

        for round in 0..24 {
            let mut c: Vec<U64Register> = (0..5)
                .map(|_| builder.constant(&u64_to_le_field_bytes(0)))
                .collect();
            let mut d: Vec<U64Register> = (0..5)
                .map(|_| builder.constant(&u64_to_le_field_bytes(0)))
                .collect();

            for y in 0..5 {
                for x in 0..5 {
                    c[x] = builder.xor(&c[x], &state[x + y * 5]);
                }
            }

            // D[x] = C[x-1] xor rot(C[x+1],1), for x in 0…4
            for x in 0..5 {
                let d_temp = builder.rotate_left(c[(x + 1) % 5], 1);
                d[x] = builder.xor(&c[(x + 4) % 5], &d_temp);
            }

            // A[x,y] = A[x,y] xor D[x], for (x,y) in (0…4,0…4)
            for y in 0..5 {
                for x in 0..5 {
                    state[x + y * 5] = builder.xor(&state[x + y * 5], &d[x]);
                }
            }

            // ############################################
            // Rho
            // ############################################

            let mut rho_x = 0;
            let mut rho_y = 1;

            for _ in 0..24 {
                // Rotate each lane by an offset
                let index = rho_x + 5 * rho_y;
                state[index] = builder.rotate_left(state[index], RHO_OFFSETS[rho_y][rho_x] % 64);
                let rho_x_prev = rho_x;
                rho_x = rho_y;
                rho_y = (2 * rho_x_prev + 3 * rho_y) % 5;
            }

            // ############################################
            // Pi
            // ############################################

            let state_cloned = state.clone();
            // B[y,2*x+3*y] = rot(A[x,y], r[x,y]), for (x,y) in (0…4,0…4)
            for y in 0..5 {
                for x in 0..5 {
                    let index = ((x + 3 * y) % 5) + x * 5;
                    state[x + y * 5] = state_cloned[index];
                }
            }

            // ############################################
            // Chi
            // ############################################

            let state_cloned = state.clone();
            // A[x,y] = B[x,y] xor ((not B[x+1,y]) and B[x+2,y]), for (x,y) in (0…4,0…4)
            for y in 0..5 {
                for x in 0..5 {
                    let index = x + y * 5;
                    let mut temp = builder.not(state_cloned[(x + 1) % 5 + y * 5]);
                    temp = builder.and(&temp, &state_cloned[(x + 2) % 5 + y * 5]);
                    state[index] = builder.xor(&state_cloned[index], &temp);
                }
            }

            // ############################################
            // Iota
            // ############################################

            // A[0,0] = A[0,0] xor RC
            state[0] = builder.xor(&state[0], &round_constant_values.get(round));
        }

        let state_output = builder.alloc_array::<U64Register>(NUM_WORDS);

        for i in 0..NUM_WORDS {
            builder.set_to_expression(&state_output.get(i), state[i].expr());
            // builder
            //     .set_to_expression_transition(&state_input.get(i).next(), &state_output(i).expr());
        }

        state_output
    }

    fn hash(builder: &mut BytesBuilder<L>, input: &[u8]) -> [u8; 32] {
        const OUTPUT_LEN: usize = 256;
        const CAPACITY: usize = OUTPUT_LEN * 2;
        const STATE_WIDTH: usize = 1600;
        const RATE: usize = STATE_WIDTH - CAPACITY;

        fn pad(n_bytes: usize) -> Vec<u8> {
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

        if input.len() > (RATE / 8) {
            panic!("Sponge not supported");
        };

        // Padding
        let pad_vec: Vec<u8> = pad(input.len());
        let mut p = input.to_vec();
        p.extend(pad_vec);

        // Transform padded input into array of words
        let mut words = to_words(&p);
        words.resize(25, 0);
        let state: [u64; 25] = words.try_into().unwrap();

        let air_state = builder.constant_array(&state.map(u64_to_le_field_bytes));
        // The permutation
        let new_air_state = Self::keccak_p(builder, air_state);

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
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::util::timing::TimingTree;
    use serde::{Deserialize, Serialize};
    use tiny_keccak::{Hasher, Keccak};

    use crate::chip::builder::tests::GoldilocksCubicParameters;
    use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::uint::register::U64Register;
    use crate::chip::uint::util::{u64_from_le_field_bytes, u64_to_le_field_bytes};
    use crate::chip::AirParameters;
    use crate::machine::builder::Builder;
    use crate::machine::bytes::builder::BytesBuilder;
    use crate::machine::hash::keccak::{Keccak256, KeccakAir, KeccakPure};
    use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Keccak256Test;

    impl AirParameters for Keccak256Test {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 77577;
        const EXTENDED_COLUMNS: usize = 46086;
    }

    #[test]
    fn test_pure_keccak256() {
        let input = (0..32).collect::<Vec<u8>>();

        // Compare with tiny-keccak
        let mut ref_hasher = Keccak::v256();
        ref_hasher.update(&input);
        let mut expected = [0u8; 32];
        ref_hasher.finalize(&mut expected);

        let result = KeccakPure::keccak256(&input);
        assert_eq!(&result, &expected);
    }
    #[test]
    fn test_air_keccak_p() {
        type C = CurtaPoseidonGoldilocksConfig;
        type _Config = <C as CurtaConfig<2>>::GenericConfig;

        const NUM_WORDS: usize = 25;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_air_keccak_p", log::Level::Debug);

        let mut builder = BytesBuilder::<Keccak256Test>::new();

        let state = builder.constant_array(&[u64_to_le_field_bytes(0); NUM_WORDS]);
        let new_state = Keccak256::keccak_p(&mut builder, state);

        let num_rows = 1 << 3;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);
        // NOTE: you always need to write something to the trace even if you have zero values, otherwise the lookup argument will fail at the proof level
        writer.write_global_instructions(&stark.air_data);

        for i in 0..num_rows {
            writer.write_row_instructions(&stark.air_data, i);
            // println!("{:?}", writer.read_array::<U64Register, 25>(&new_state, i));
        }

        // Check against pure vanilla implementation of keccak permutation
        let air_hash_output = writer.read_array::<U64Register, 25>(&new_state, num_rows - 1);
        let mut pure_state = [0; NUM_WORDS];
        KeccakPure::keccak_p(&mut pure_state);
        assert_eq!(
            pure_state,
            air_hash_output.map(|i| u64_from_le_field_bytes(&i))
        );

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();
    }

    #[test]
    fn test_air_keccak256() {}
}
