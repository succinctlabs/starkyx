use core::fmt::Debug;

use log::debug;
use num::{Num, Zero};
use plonky2::util::log2_ceil;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::chip::instruction::empty;
// use super::data::SHAData;
use crate::chip::memory::time::Time;
use crate::chip::memory::value::MemoryValue;
use crate::chip::register::array::ArrayRegister;
use crate::chip::register::bit::BitRegister;
use crate::chip::register::element::ElementRegister;
use crate::chip::register::{Register, RegisterSerializable};
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::uint::register::U64Register;
use crate::chip::uint::util::{u32_to_le_field_bytes, u64_to_le_field_bytes};
use crate::chip::AirParameters;
use crate::machine::builder::ops::Xor;
use crate::machine::builder::Builder;
use crate::machine::bytes::builder::BytesBuilder;
// use crate::machine::hash::sha::data::{SHAMemory, SHAPublicData, SHATraceData};
use crate::math::prelude::*;

const DUMMY_INDEX: u64 = i32::MAX as u64;

use crate::chip::memory::pointer::slice::Slice;

// TODO: refactor this later into data.rs file
pub struct KeccakData<T, const LENGTH: usize> {
    pub public: KeccakPublicData<T>,
    pub trace: KeccakTraceData<LENGTH>,
    pub memory: KeccakMemory<T>,
    pub degree: usize,
}

pub struct KeccakPublicData<T> {
    pub initial_hash: ArrayRegister<T>,
    pub padded_chunks: Vec<ArrayRegister<T>>,
    pub digest_indices: ArrayRegister<ElementRegister>,
}

pub struct KeccakTraceData<const LENGTH: usize> {
    pub(crate) is_preprocessing: BitRegister,
    pub(crate) process_id: ElementRegister,
    pub(crate) cycle_end_bit: BitRegister,
    pub index: ElementRegister,
    pub is_dummy: BitRegister,
}

pub struct KeccakMemory<T> {
    pub(crate) round_constants: Slice<T>,
    pub(crate) w: Slice<T>,
    pub shift_read_mult: Slice<ElementRegister>,
    pub end_bit: Slice<BitRegister>,
    pub digest_bit: Slice<BitRegister>,
    pub dummy_index: ElementRegister,
}

/// Pure Keccak algorithm implementation.
///
/// An interface for the Keccak algorithm as a Rust function operating on numerical values.
pub trait KeccakPure<const CYCLE_LENGTH: usize>:
    Debug + Clone + 'static + Serialize + DeserializeOwned + Send + Sync
{
    type Integer: Num + Copy + Debug;

    const ROUNDS: usize = 24;
    const OUTPUT_LEN: usize = 256;
    const CAPACITY: usize = OUTPUT_LEN * 2;
    const STATE_WIDTH: usize = 1600;
    const RATE: usize = STATE_WIDTH - CAPACITY;

    // Table 2 of https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
    const RHO_OFFSETS: [[u32; 5]; 5] = [
        [0, 1, 190, 28, 91],
        [36, 300, 6, 55, 276],
        [3, 10, 171, 153, 231],
        [105, 45, 15, 21, 136],
        [210, 66, 253, 120, 78],
    ];

    // Copied from https://github.com/debris/tiny-keccak/blob/master/src/keccakf.rs
    // const RC: [u64; ROUNDS] = [
    //     1u64,
    //     0x8082u64,
    //     0x800000000000808au64,
    //     0x8000000080008000u64,
    //     0x808bu64,
    //     0x80000001u64,
    //     0x8000000080008081u64,
    //     0x8000000000008009u64,
    //     0x8au64,
    //     0x88u64,
    //     0x80008009u64,
    //     0x8000000au64,
    //     0x8000808bu64,
    //     0x800000000000008bu64,
    //     0x8000000000008089u64,
    //     0x8000000000008003u64,
    //     0x8000000000008002u64,
    //     0x8000000000000080u64,
    //     0x800au64,
    //     0x800000008000000au64,
    //     0x8000000080008081u64,
    //     0x8000000000008080u64,
    //     0x80000001u64,
    //     0x8000000080008008u64,
    // ];

    fn to_words(bytes: &[u8]) -> Vec<u64>;

    fn pad(n_bytes: usize) -> Vec<u8>;

    fn keccak256(input: &[u8]) -> [u8; 32];

    fn keccak_p(state: &mut [u64; 25]);
}

/// Keccak algorithm AIR implementation.
///
/// An interface for the Keccak algorithm as an AIR.
pub trait KeccakAir2<L: AirParameters>
where
    L::Instruction: UintInstructions,
{
    const ROUNDS: usize = 24;
    const OUTPUT_LEN: usize = 256;
    const CAPACITY: usize = OUTPUT_LEN * 2;
    const STATE_WIDTH: usize = 1600;
    const RATE: usize = STATE_WIDTH - CAPACITY;

    // Table 2 of https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
    const RHO_OFFSETS: [[u32; 5]; 5] = [
        [0, 1, 190, 28, 91],
        [36, 300, 6, 55, 276],
        [3, 10, 171, 153, 231],
        [105, 45, 15, 21, 136],
        [210, 66, 253, 120, 78],
    ];

    // Copied from https://github.com/debris/tiny-keccak/blob/master/src/keccakf.rs
    const RC: [u64; ROUNDS] = [
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

    fn to_words(builder: &mut BytesBuilder<L>, bytes: &[u8]) -> Vec<u64>;

    fn pad(builder: &mut BytesBuilder<L>, n_bytes: usize) -> Vec<u8>;

    fn keccak256(builder: &mut BytesBuilder<L>, input: &[u8]) -> [u8; 32];

    fn keccak_p(builder: &mut BytesBuilder<L>, state: ArrayRegister<U64Register>) {
        for i in 0..ROUNDS {
            // TODO: use the i counter to read the variables in the register with the timestamp
            // ############################################
            // Theta
            // ############################################

            // let mut c = [0; 5];
            // let mut d = [0; 5];
            let zero_entry =
                builder.constant::<U64Register>(&[L::Field::from_canonical_usize(0); 8]);
            let c = builder.uninit_slice();
            let d = builder.uninit_slice();
            let zero = builder.constant(&L::Field::from_canonical_usize(0));

            for i in 0..5 {
                // ??? Is there another way to produce a zero constant
                builder.store(&c.get(i), zero_entry, &Time::zero(), Some(zero));
                builder.store(&d.get(i), zero_entry, &Time::zero(), Some(zero));
            }
            // TODO:  ??? what is the best way to initialize working variables? alloc_array or through the load
            // let c = builder.alloc_array::<U64Register>(5);
            // let d = builder.alloc_array::<U64Register>(5);

            // C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4], for x in 0…4
            for y in 0..5 {
                for x in 0..5 {
                    let c_val = builder.load::<U64Register>(&c.get(x), &Time::zero());
                    builder.xor(c_val, state.get(x + y * 5));

                    // c[x] ^= state[x + y * 5];
                }
            }

            /*
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
                            state[index] = state[index].rotate_left(RHO_OFFSETS[rho_y][rho_x] % 64);
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
                        state[0] ^= RC[i];
            let state_next;

            for ((s, s_next), s_init) in state.iter().zip(state_next).zip(initial) {
                builder.select_next(end_bit, s_init, s_next, s);
            }
            */
        }
    }

    fn data(builder: &mut BytesBuilder<L>, words: &[ArrayRegister<U64Register>]) {
        //
        // Store Keccak Round Constants RC
        //
        let num_round_element: ElementRegister =
            builder.constant(&L::Field::from_canonical_usize(ROUNDS));

        // Initialize the round constants and set them to the constant value.
        let round_constant_values =
            builder.constant_array::<U64Register>(&Self::RC.map(u64_to_le_field_bytes));

        // Store the round constants in a slice to be able to load them in the trace.
        let round_constants = builder.uninit_slice();

        for i in 0..ROUNDS {
            builder.store(
                &round_constants.get(i),
                round_constant_values.get(i),
                &Time::zero(),
                Some(num_round_element),
            )
        }
        //
        // Store Keccak Rotation Offsets RHO
        //
        let num_rho_element: ElementRegister =
            builder.constant(&L::Field::from_canonical_usize(25));

        // let rho_constant_values = builder.constant_array::<U64Register>(
        //     &Self::RHO_OFFSETS
        //         .into_iter()
        //         .flat_map(|v| v.into_iter())
        //         .collect::<Vec<u32>>() // Add explicit type annotation here
        //         .into_iter() // Convert the vector into an iterator
        //         .map(u32_to_le_field_bytes())
        //         .collect::<Vec<_>>(), // Collect the results into a vector
        // );
        // Store the round constants in a slice to be able to load them in the trace.
        // let rho_constants = builder.uninit_slice();

        // for i in 0..25 {
        //     builder.store(
        //         &rho_constants.get(i),
        //         rho_constant_values.get(i),
        //         &Time::zero(),
        //         Some(num_rho_element),
        //     )
        // }
        //
        // TODO: Store witness in words format
        //
        // let witness = builder.uninit_slice();
    }
}

// Keccak256 parameters in bits
const ROUNDS: usize = 24;
const OUTPUT_LEN: usize = 256;
const CAPACITY: usize = OUTPUT_LEN * 2;
const STATE_WIDTH: usize = 1600;
const RATE: usize = STATE_WIDTH - CAPACITY;

// Table 2 of https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 1, 190, 28, 91],
    [36, 300, 6, 55, 276],
    [3, 10, 171, 153, 231],
    [105, 45, 15, 21, 136],
    [210, 66, 253, 120, 78],
];

// Copied from https://github.com/debris/tiny-keccak/blob/master/src/keccakf.rs
const RC: [u64; ROUNDS] = [
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
    let padded_size_in_bytes = n_bytes.next_multiple_of(RATE / 8);
    let mut padding = vec![0u8; padded_size_in_bytes - n_bytes];
    let padding_len = padding.len();
    padding[0] = 0x01;
    padding[padding_len - 1] = 0x80;
    padding
}

pub fn keccak256(input: &[u8]) -> [u8; 32] {
    if input.len() > (RATE / 8) {
        panic!("Sponge not supported");
    };

    // Padding
    let pad_vec = pad(input.len());
    let mut p = input.to_vec();
    p.extend(pad_vec);

    // Transform padded input into array of words
    let mut words = to_words(&p);
    words.resize(25, 0);
    let mut state: [u64; 25] = words.try_into().unwrap();

    // The permutation
    keccak_p(&mut state);

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
    for i in 0..ROUNDS {
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
            state[index] = state[index].rotate_left(RHO_OFFSETS[rho_y][rho_x] % 64);
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
        state[0] ^= RC[i];
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::util::timing::{self, TimingTree};
    use plonky2_maybe_rayon::IndexedParallelIterator;
    use rand::Rng;
    use serde::Deserialize;
    use tiny_keccak::{Hasher, Keccak};

    use super::*;
    use crate::chip::builder::tests::GoldilocksCubicParameters;
    use crate::chip::builder::{self};
    use crate::chip::instruction::set::AirInstruction;
    use crate::chip::trace::writer::{InnerWriterData, TraceWriter};
    use crate::chip::uint::operations::instruction::UintInstruction;
    use crate::chip::uint::register::ByteArrayRegister;
    use crate::plonky2::stark::config::{CurtaConfig, CurtaPoseidonGoldilocksConfig};

    #[test]
    fn test_permutation() {
        let mut input = [0; 25];
        keccak_p(&mut input);
        println!("output {:?}", input[0]);
        println!(
            "output Field {:?}",
            GoldilocksField::from_canonical_u64(input[0])
        );
    }
    #[test]
    fn test_keccak_p() {
        let input = (0..32).collect::<Vec<u8>>();

        // Compare with tiny-keccak
        let mut ref_hasher = Keccak::v256();
        ref_hasher.update(&input);
        let mut expected = [0u8; 32];
        ref_hasher.finalize(&mut expected);

        let result = keccak256(&input);
        assert_eq!(&result, &expected);
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ByteSliceMemTest;

    impl AirParameters for ByteSliceMemTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_FREE_COLUMNS: usize = 18;
        const EXTENDED_COLUMNS: usize = 24;
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ByteTest;

    impl AirParameters for ByteTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        type Instruction = UintInstruction;

        const NUM_ARITHMETIC_COLUMNS: usize = 0;
        const NUM_FREE_COLUMNS: usize = 225;
        const EXTENDED_COLUMNS: usize = 126;
    }

    #[test]
    fn test_air_keccak256() {
        type C = CurtaPoseidonGoldilocksConfig;
        type Config = <C as CurtaConfig<2>>::GenericConfig;

        const ROUNDS: usize = 24;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut timing = TimingTree::new("test_byte_multi_stark", log::Level::Debug);

        let mut builder = BytesBuilder::<ByteTest>::new();

        const RC: [u64; ROUNDS] = [
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

        let num_round_element: ElementRegister =
            builder.constant(&Field::from_canonical_usize(ROUNDS));

        // Initialize the round constants and set them to the constant value.
        let round_constant_values =
            builder.constant_array::<U64Register>(&RC.map(u64_to_le_field_bytes));

        // Store the round constants in a slice to be able to load them in the trace.
        let round_constants = builder.uninit_slice();

        let num_digests = 16;
        let hash_state_public = (0..num_digests)
            .map(|_| {
                let r = builder.alloc_public::<U64Register>();
                // builder.set_to_expression(&r, Field::ZERO.into());
            })
            .collect::<Vec<_>>();

        for i in 0..ROUNDS {
            builder.store(
                &round_constants.get(i),
                round_constant_values.get(i),
                &Time::zero(),
                Some(num_round_element),
            );
            builder.load(&round_constants.get(i), &Time::zero());
        }

        let a = builder.alloc::<U64Register>();
        let b = builder.alloc::<U64Register>();

        let num_ops = 1;
        for _ in 0..num_ops {
            let _ = builder.and(&a, &b);
        }

        let num_rows = 1 << 4;
        let stark = builder.build::<C, 2>(num_rows);

        let writer = TraceWriter::new(&stark.air_data, num_rows);

        // let mut rng = rand::thread_rng();
        // for i in 0..num_rows {
        //     let a_val = rng.gen::<u64>();
        //     let b_val = rng.gen::<u64>();
        //     writer.write(&a, &u64_to_le_field_bytes(a_val), i);
        //     writer.write(&b, &u64_to_le_field_bytes(b_val), i);
        //     writer.write_row_instructions(&stark.air_data, i);
        //     println!("writer row_idx {}: {:?}", i, writer.read(&a, i));
        // }

        let InnerWriterData { trace, public, .. } = writer.into_inner().unwrap();
        let proof = stark.prove(&trace, &public, &mut timing).unwrap();

        stark.verify(proof.clone(), &public).unwrap();
    }
}
