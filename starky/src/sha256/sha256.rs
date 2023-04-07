//! Educational implementation to build helper functions needed for building the trace
//! table for the SHA256 STARK. Instead of using byte or u32-level bitwise ops, this implementation
//! purposely does everything bitwise, as that is how it will be needed for the STARK.

use crate::sha256::helper::{
    add2, add4, add5, and2, not, rotate, shr, usize_to_be_bits, xor2, xor3, ONE, ZERO,
};

// First 32 bits of the fractional parts of the square roots of the first 8 primes.
// Reference: https://en.wikipedia.org/wiki/SHA-2
const H: [usize; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

// First 32 bits of the fractional parts of the cube roots of the first 64 primes.
// Reference: https://en.wikipedia.org/wiki/SHA-2
const K: [usize; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const U64_BIT_LENGTH: usize = 64;
const SHA256_INPUT_LENGTH: usize = 512;
const SHA256_DIGEST_LENGTH: usize = 256;

pub fn sha256(input_message: [bool; SHA256_INPUT_LENGTH]) -> [bool; SHA256_DIGEST_LENGTH] {
    // The length-encoded message length ("L + 1 + 64").
    const SEPERATOR_LENGTH: usize = 1;
    const U64_LENGTH: usize = 64;
    const ENCODED_MESSAGE_LENGTH: usize = SHA256_INPUT_LENGTH + SEPERATOR_LENGTH + U64_LENGTH;

    // The multiple of 512-bit padded message length. Padding length is "K".
    const REMAINDER_LENGTH: usize = ENCODED_MESSAGE_LENGTH % 512;
    const PADDING_LENGTH: usize = if REMAINDER_LENGTH == 0 {
        0
    } else {
        512 - REMAINDER_LENGTH
    };
    const PADDED_MESSAGE_LENGTH: usize = ENCODED_MESSAGE_LENGTH + PADDING_LENGTH;

    // Initialization of core variables.
    let mut padded_message = [ZERO; PADDED_MESSAGE_LENGTH];

    // Begin with the original message of length "L".
    for i in 0..SHA256_INPUT_LENGTH {
        padded_message[i] = input_message[i];
    }

    // Append a single '1' bit.
    padded_message[SHA256_INPUT_LENGTH] = ONE;

    // Append L as a 64-bit big-endian integer.
    let sha256_input_length_bits = usize_to_be_bits::<U64_BIT_LENGTH>(SHA256_INPUT_LENGTH);
    for i in 0..U64_BIT_LENGTH {
        padded_message[SHA256_INPUT_LENGTH + i + 1 + PADDING_LENGTH] = sha256_input_length_bits[i];
    }

    // At this point, the padded message should be of the following form.
    //      <message of length L> 1 <K zeros> <L as 64 bit integer>
    // Now, we will process the padded message in 512 bit chunks and begin referring to the
    // padded message as "message".
    const SHA256_CHUNK_LENGTH: usize = 512;
    const SHA256_WORD_LENGTH: usize = 32;
    const SHA256_MESSAGE_SCHEDULE_ARRAY_LENGTH: usize = 64;

    let message = padded_message;
    let num_chunks = message.len() / SHA256_CHUNK_LENGTH;

    let mut h = H
        .into_iter()
        .map(|x| usize_to_be_bits::<SHA256_WORD_LENGTH>(x))
        .collect::<Vec<_>>();
    let k = K
        .into_iter()
        .map(|x| usize_to_be_bits::<SHA256_WORD_LENGTH>(x))
        .collect::<Vec<_>>();

    for i in 0..num_chunks {
        // The 64-entry message schedule array of 32-bit words.
        let mut w = [[ZERO; SHA256_WORD_LENGTH]; SHA256_MESSAGE_SCHEDULE_ARRAY_LENGTH];

        // Copy chunk into first 16 words w[0..15] of the message schedule array.
        let chunk_offset = i * SHA256_CHUNK_LENGTH;
        for j in 0..16 {
            let word_offset = j * 32;
            for k in 0..32 {
                w[j][k] = message[chunk_offset + word_offset + k];
            }
        }

        // Extend the first 16 words into the remaining 48 words w[16..63].
        for j in 16..SHA256_MESSAGE_SCHEDULE_ARRAY_LENGTH {
            let s0 = xor3(
                rotate(w[j - 15], 7),
                rotate(w[j - 15], 18),
                shr(w[j - 15], 3),
            );
            // println!("{} {}", j, be_bits_to_usize(s0));
            let s1 = xor3(
                rotate(w[j - 2], 17),
                rotate(w[j - 2], 19),
                shr(w[j - 2], 10),
            );
            w[j] = add4(w[j - 16], s0, w[j - 7], s1);
        }

        let (mut sa, mut sb, mut sc, mut sd, mut se, mut sf, mut sg, mut sh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        const NUM_COMPRESSION_ROUNDS: usize = 64;
        for j in 0..NUM_COMPRESSION_ROUNDS {
            let s1 = xor3(rotate(se, 6), rotate(se, 11), rotate(se, 25));
            let ch = xor2(and2(se, sf), and2(not(se), sg));
            let temp1 = add5(sh, s1, ch, k[j], w[j]);
            let s0 = xor3(rotate(sa, 2), rotate(sa, 13), rotate(sa, 22));
            let maj = xor3(and2(sa, sb), and2(sa, sc), and2(sb, sc));
            let temp2 = add2(s0, maj);

            sh = sg;
            sg = sf;
            sf = se;
            se = add2(sd, temp1);
            sd = sc;
            sc = sb;
            sb = sa;
            sa = add2(temp1, temp2);
        }

        h[0] = add2(h[0], sa);
        h[1] = add2(h[1], sb);
        h[2] = add2(h[2], sc);
        h[3] = add2(h[3], sd);
        h[4] = add2(h[4], se);
        h[5] = add2(h[5], sf);
        h[6] = add2(h[6], sg);
        h[7] = add2(h[7], sh);
    }

    let mut digest = [ZERO; SHA256_DIGEST_LENGTH];
    for i in 0..h.len() {
        for j in 0..SHA256_WORD_LENGTH {
            digest[i * SHA256_WORD_LENGTH + j] = h[i][j];
        }
    }

    digest
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;
    use crate::sha256::helper::be_bytes_to_bits;
    use crate::sha256::reference::Sha256;

    const SHA256_INPUT_BYTE_LENGTH: usize = SHA256_INPUT_LENGTH / 8;

    fn test(input: [u8; SHA256_INPUT_BYTE_LENGTH]) {
        let input_bits = be_bytes_to_bits(input);
        let expected_digest = be_bytes_to_bits(Sha256::digest(&input[..]));
        let digest = sha256(input_bits).to_vec();
        for i in 0..expected_digest.len() {
            assert_eq!(expected_digest[i], digest[i]);
        }
    }

    #[test]
    fn test_sha256_zeros() {
        let input = [0 as u8; 64];
        test(input);
    }

    #[test]
    fn test_sha256_ones() {
        let input = [1 as u8; 64];
        test(input);
    }

    #[test]
    fn test_sha256_random() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let input = [rng.gen::<u8>(); 64];
            test(input);
        }
    }
}
