use super::util::SHA512Util;
use super::{INITIAL_HASH, ROUND_CONSTANTS, SHA512};
use crate::machine::hash::sha::algorithm::SHAPure;

impl SHAPure<80> for SHA512 {
    type Integer = u64;

    const INITIAL_HASH: [Self::Integer; 8] = INITIAL_HASH;
    const ROUND_CONSTANTS: [Self::Integer; 80] = ROUND_CONSTANTS;

    fn pad(msg: &[u8]) -> Vec<u8> {
        SHA512Util::pad(msg)
    }

    fn pre_process(chunk: &[u8]) -> [Self::Integer; 80] {
        SHA512Util::pre_process(chunk)
    }

    fn process(
        hash: [Self::Integer; 8],
        w: &[Self::Integer; 80],
        round_constants: [Self::Integer; 80],
    ) -> [Self::Integer; 8] {
        SHA512Util::process(hash, w, round_constants)
    }

    fn decode(digest: &str) -> [Self::Integer; 8] {
        SHA512Util::decode(digest)
    }
}
