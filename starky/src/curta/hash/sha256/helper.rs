pub const ZERO: bool = false;
pub const ONE: bool = true;

// Converts a usize to a list of bits in big endian order.
pub fn usize_to_be_bits<const L: usize>(x: usize) -> [bool; L] {
    let mut bits = [ZERO; L];
    let mut sum: usize = 0;
    for i in 0..L {
        let digit: usize = 1 << (L - i - 1);
        bits[i] = x & digit != 0;
        sum += (bits[i] as usize) * digit;
    }
    assert!(sum == x);
    bits
}

pub fn be_bits_to_usize<const L: usize>(bits: [bool; L]) -> usize {
    let mut sum: usize = 0;
    for i in 0..L {
        let digit: usize = 1 << (L - i - 1);
        sum += (bits[i] as usize) * digit;
    }
    sum
}

pub fn not<const L: usize>(x: [bool; L]) -> [bool; L] {
    let mut y = [ZERO; L];
    for i in 0..L {
        y[i] = !x[i];
    }
    y
}

pub fn and2<const L: usize>(x: [bool; L], y: [bool; L]) -> [bool; L] {
    let mut z = [ZERO; L];
    for i in 0..L {
        z[i] = x[i] & y[i];
    }
    z
}

pub fn xor2<const L: usize>(x: [bool; L], y: [bool; L]) -> [bool; L] {
    let mut z = [ZERO; L];
    for i in 0..L {
        z[i] = x[i] ^ y[i];
    }
    z
}

pub fn xor3<const L: usize>(x: [bool; L], y: [bool; L], z: [bool; L]) -> [bool; L] {
    let mut res = [ZERO; L];
    for i in 0..L {
        res[i] = x[i] ^ y[i] ^ z[i];
    }
    res
}

pub fn add2<const L: usize>(a: [bool; L], b: [bool; L]) -> [bool; L] {
    let sum = (be_bits_to_usize(a) + be_bits_to_usize(b)) % (1 << L);
    usize_to_be_bits(sum)
}

pub fn add4<const L: usize>(a: [bool; L], b: [bool; L], c: [bool; L], d: [bool; L]) -> [bool; L] {
    let sum =
        (be_bits_to_usize(a) + be_bits_to_usize(b) + be_bits_to_usize(c) + be_bits_to_usize(d))
            % (1 << L);
    usize_to_be_bits(sum)
}

pub fn add5<const L: usize>(
    a: [bool; L],
    b: [bool; L],
    c: [bool; L],
    d: [bool; L],
    e: [bool; L],
) -> [bool; L] {
    let sum = (be_bits_to_usize(a)
        + be_bits_to_usize(b)
        + be_bits_to_usize(c)
        + be_bits_to_usize(d)
        + be_bits_to_usize(e))
        % (1 << L);
    usize_to_be_bits(sum)
}

pub fn rotate<const L: usize>(x: [bool; L], offset: usize) -> [bool; L] {
    let mut y = [ZERO; L];
    for i in 0..L {
        y[(i + offset) % L] = x[i];
    }
    y
}

pub fn shr<const L: usize>(x: [bool; L], offset: usize) -> [bool; L] {
    let mut y = [ZERO; L];
    for i in 0..L {
        y[i] = if i < offset { ZERO } else { x[i - offset] }
    }
    y
}

pub fn be_bytes_to_bits<const L: usize>(msg: [u8; L]) -> [bool; L * 8] {
    let mut res = [ZERO; L * 8];
    for i in 0..msg.len() {
        let char = msg[i];
        for j in 0..8 {
            if (char & (1 << (7 - j))) != 0 {
                res[i * 8 + j] = ONE;
            } else {
                res[i * 8 + j] = ZERO;
            }
        }
    }
    res
}

pub fn format_bits(x: Vec<bool>) -> Vec<u32> {
    x.into_iter().map(|x| x as u32).collect()
}
