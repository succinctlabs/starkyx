use super::{BLAKE2B, COMPRESS_IV, STATE_SIZE, WORK_VECTOR_SIZE};
use crate::machine::hash::blake::blake2b::SIGMA_PERMUTATIONS;
use crate::machine::hash::HashPureInteger;

impl HashPureInteger for BLAKE2B {
    type Integer = u64;
}

pub trait BLAKE2BPure: HashPureInteger {
    fn compress(
        msg_chunk: &[u8],
        state: &mut [Self::Integer; STATE_SIZE],
        bytes_compressed: u64,
        last_chunk: bool,
    ) -> [Self::Integer; STATE_SIZE];

    fn mix(
        v: &mut [Self::Integer; WORK_VECTOR_SIZE],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        x: Self::Integer,
        y: Self::Integer,
    );
}

impl BLAKE2BPure for BLAKE2B {
    fn compress(
        msg_chunk: &[u8],
        state: &mut [Self::Integer; STATE_SIZE],
        bytes_compressed: u64,
        last_chunk: bool,
    ) -> [Self::Integer; STATE_SIZE] {
        // Set up the work vector V
        let mut v: [Self::Integer; WORK_VECTOR_SIZE] = [0; WORK_VECTOR_SIZE];

        v[..8].copy_from_slice(&state[..STATE_SIZE]);
        v[8..16].copy_from_slice(&COMPRESS_IV);

        v[12] ^= bytes_compressed;
        if last_chunk {
            v[14] ^= 0xFFFFFFFFFFFFFFFF;
        }

        let msg_u64_chunks = msg_chunk
            .chunks_exact(8)
            .map(|x| Self::Integer::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();

        for s in SIGMA_PERMUTATIONS.iter() {
            Self::mix(
                &mut v,
                0,
                4,
                8,
                12,
                msg_u64_chunks[s[0] as usize],
                msg_u64_chunks[s[1] as usize],
            );
            Self::mix(
                &mut v,
                1,
                5,
                9,
                13,
                msg_u64_chunks[s[2] as usize],
                msg_u64_chunks[s[3] as usize],
            );
            Self::mix(
                &mut v,
                2,
                6,
                10,
                14,
                msg_u64_chunks[s[4] as usize],
                msg_u64_chunks[s[5] as usize],
            );
            Self::mix(
                &mut v,
                3,
                7,
                11,
                15,
                msg_u64_chunks[s[6] as usize],
                msg_u64_chunks[s[7] as usize],
            );

            Self::mix(
                &mut v,
                0,
                5,
                10,
                15,
                msg_u64_chunks[s[8] as usize],
                msg_u64_chunks[s[9] as usize],
            );
            Self::mix(
                &mut v,
                1,
                6,
                11,
                12,
                msg_u64_chunks[s[10] as usize],
                msg_u64_chunks[s[11] as usize],
            );
            Self::mix(
                &mut v,
                2,
                7,
                8,
                13,
                msg_u64_chunks[s[12] as usize],
                msg_u64_chunks[s[13] as usize],
            );
            Self::mix(
                &mut v,
                3,
                4,
                9,
                14,
                msg_u64_chunks[s[14] as usize],
                msg_u64_chunks[s[15] as usize],
            );
        }

        for i in 0..STATE_SIZE {
            state[i] ^= v[i];
        }

        for i in 0..STATE_SIZE {
            state[i] ^= v[i + 8];
        }

        *state
    }

    fn mix(
        v: &mut [Self::Integer; WORK_VECTOR_SIZE],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        x: Self::Integer,
        y: Self::Integer,
    ) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }
}
