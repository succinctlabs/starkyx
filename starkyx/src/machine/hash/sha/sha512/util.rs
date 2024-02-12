pub struct SHA512Util;

impl SHA512Util {
    pub fn pad(msg: &[u8]) -> Vec<u8> {
        let mut padded_msg = Vec::new();
        padded_msg.extend_from_slice(msg);
        padded_msg.push(1 << 7);

        // Find number of zeros
        let mdi = msg.len() % 128;
        assert!(mdi < 240);
        let padlen = if mdi < 112 { 112 - mdi } else { 240 - mdi };
        // Pad with zeros
        padded_msg.extend_from_slice(&vec![0u8; padlen]);

        // add length as 64 bit number
        let len = ((msg.len() * 8) as u128).to_be_bytes();
        padded_msg.extend_from_slice(&len);

        padded_msg
    }

    pub fn pre_process(chunk: &[u8]) -> [u64; 80] {
        let chunk_u64 = chunk
            .chunks_exact(8)
            .map(|x| u64::from_be_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut w = [0u64; 80];

        w[..16].copy_from_slice(&chunk_u64[..16]);

        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        w
    }

    pub fn decode(digest: &str) -> [u64; 8] {
        hex::decode(digest)
            .unwrap()
            .chunks_exact(8)
            .map(|x| u64::from_be_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}
