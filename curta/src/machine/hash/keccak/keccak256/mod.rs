use serde::{Deserialize, Serialize};

pub(crate) const KECCAK_ROUNDS: usize = 24;
// The round constants RC[i] are given in the table below for the maximum lane size 64. For smaller sizes, they are simply truncated.
// We could also generate the round constants with appropriate formula, the linear feedback shift register (LFSR) (see [Keccak Reference, Section 1.2]).
pub(crate) const ROUND_CONSTANTS: [u64; KECCAK_ROUNDS] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

pub(crate) const ROTATION_OFFSETS: [[u8; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

const RHO: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

const PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

const SPONGE_RATE: usize = 1088;
const SPONGE_CAPACITY: usize = 512;
const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;
const SPONGE_WORDS: usize = 25;
const DELIM: u8 = 0x06;
const OUTPUT_BYTE_LEN: usize = 32;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Keccak256;

impl Keccak256 {
    // Inputs for Keccak(rate, capacity, inputBytes, delimitedSuffix, outputByteLen):
    // For 256 bits use Keccak(1088, 512, inputBytes, 0x06, 256//8)
    fn state_permute(state: &mut [u64; SPONGE_WORDS]) -> Option<[u64; 25]> {
        // In Keccak, the underlying function is a permutation chosen in a set of seven Keccak-f permutations,
        // denoted Keccak-f[b], where b∈{25,50,100,200,400,800,1600} is the width of the permutation.
        //
        // Keccak-f[b](A) {
        //   for i in 0…n-1
        //     A = Round[b](A, RC[i])
        //   return A
        // }
        // Round[b](A,RC)
        for i in 0..KECCAK_ROUNDS {
            let mut c: [u64; 5] = [0; 5];
            // # θ step
            // C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4],   for x in 0…4
            for x in 0..5 {
                for y_count in 0..5 {
                    let y = y_count * 5;
                    c[x] ^= state[x + y];
                }
            }
            // D[x] = C[x-1] xor rot(C[x+1],1),                             for x in 0…4
            // A[x,y] = A[x,y] xor D[x],                           for (x,y) in (0…4,0…4)
            for x in 0..5 {
                for y_count in 0..5 {
                    let y = y_count * 5;
                    state[y + x] ^= c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
                }
            }

            // # ρ and π steps
            // B[y,2*x+3*y] = rot(A[x,y], r[x,y]),                 for (x,y) in (0…4,0…4)
            let mut last = state[1];
            for x in 0..24 {
                c[0] = state[PI[x]];
                state[PI[x]] = last.rotate_left(RHO[x]);
                last = c[0];
            }

            // # χ step
            // A[x,y] = B[x,y] xor ((not B[x+1,y]) and B[x+2,y]),  for (x,y) in (0…4,0…4)
            for y_step in 0..5 {
                let y = y_step * 5;

                // for x in 0..5 {
                //     c[x] = state[y + x];
                // }
                // below code is more efficient than the one above
                c.copy_from_slice(&state[y..y + 5]);

                for x in 0..5 {
                    state[y + x] = c[x] ^ ((!c[(x + 1) % 5]) & (c[(x + 2) % 5]));
                }
            }

            // # ι step
            // A[0,0] = A[0,0] xor RC
            state[0] ^= ROUND_CONSTANTS[i];

            // return A
        }
        None
    }

    fn finalize(input: &[u8]) -> [u8; 32] {
        if SPONGE_WIDTH != 1600 || SPONGE_RATE % 8 != 0 {
            panic!("");
        }

        let rate_in_bytes: usize = SPONGE_RATE / 8;
        let mut block_size = 0;

        // initialization
        let mut state: Vec<u8> = vec![0; 200];
        let mut input_byte_len = input.len();

        // Absorbing phase
        while (input_byte_len > 0) {
            block_size = std::cmp::min(input_byte_len, rate_in_bytes);
            for i in 0..block_size {
                state[i] ^= input[i];
            }
            input += block_size;
            input_byte_len -= block_size;

            if (block_size == rate_in_bytes) {
                state_permute(state);
                block_size = 0;
            }
        }

        // absorbing

        // squeezing

        // # Padding
        // d = 2^|Mbits| + sum for i=0..|Mbits|-1 of 2^i*Mbits[i]
        // P = Mbytes || d || 0x00 || … || 0x00
        // P = P xor (0x00 || … || 0x00 || 0x80)

        // # Initialization
        // S[x,y] = 0,                               for (x,y) in (0…4,0…4)

        // # Absorbing phase
        // for each block Pi in P
        //   S[x,y] = S[x,y] xor Pi[x+5*y],          for (x,y) such that x+5*y < r/w
        //   S = Keccak-f[r+c](S)

        // # Squeezing phase
        // Z = empty string
        // while output is requested
        //   Z = Z || S[x,y],                        for (x,y) such that x+5*y < r/w
        //   S = Keccak-f[r+c](S)

        // return Z

        //     uint8_t state[200];
        // unsigned int rateInBytes = rate/8;
        // unsigned int blockSize = 0;
        // unsigned int i;

        // if (((rate + capacity) != 1600) || ((rate % 8) != 0))
        //     return;

        // /* === Initialize the state === */
        // memset(state, 0, sizeof(state));

        // /* === Absorb all the input blocks === */
        // while(inputByteLen > 0) {
        //     blockSize = MIN(inputByteLen, rateInBytes);
        //     for(i=0; i<blockSize; i++)
        //         state[i] ^= input[i];
        //     input += blockSize;
        //     inputByteLen -= blockSize;

        //     if (blockSize == rateInBytes) {
        //         KeccakF1600_StatePermute(state);
        //         blockSize = 0;
        //     }
        // }

        // /* === Do the padding and switch to the squeezing phase === */
        // /* Absorb the last few bits and add the first bit of padding (which coincides with the delimiter in delimitedSuffix) */
        // state[blockSize] ^= delimitedSuffix;
        // /* If the first bit of padding is at position rate-1, we need a whole new block for the second bit of padding */
        // if (((delimitedSuffix & 0x80) != 0) && (blockSize == (rateInBytes-1)))
        //     KeccakF1600_StatePermute(state);
        // /* Add the second bit of padding */
        // state[rateInBytes-1] ^= 0x80;
        // /* Switch to the squeezing phase */
        // KeccakF1600_StatePermute(state);

        // /* === Squeeze out all the output blocks === */
        // while(outputByteLen > 0) {
        //     blockSize = MIN(outputByteLen, rateInBytes);
        //     memcpy(output, state, blockSize);
        //     output += blockSize;
        //     outputByteLen -= blockSize;

        //     if (outputByteLen > 0)
        //         KeccakF1600_StatePermute(state);
        // }
        [0; 32]
    }
}
