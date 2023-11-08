pub mod keccak256;
use crate::chip::uint::operations::instruction::UintInstructions;
use crate::chip::AirParameters;

struct Keccak256;
pub trait KeccakAir<L: AirParameters>
where
    L::Instruction: UintInstructions,
{
    const ROUNDS: usize = 24;

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
}
