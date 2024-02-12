pub struct SHA256Util;

impl SHA256Util {
    pub fn decode(digest: &str) -> [u32; 8] {
        hex::decode(digest)
            .unwrap()
            .chunks_exact(4)
            .map(|x| u32::from_be_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}
