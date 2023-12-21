pub struct BLAKE2BUtil;

impl BLAKE2BUtil {
    pub fn pad(msg: &[u8], max_chunk_size: u64) -> Vec<u8> {
        let mut msg_chunk_size = msg.len() as u64 / 128;

        if (msg.len() % 128 != 0) || msg.is_empty() {
            msg_chunk_size += 1;
        }

        assert!(msg_chunk_size <= max_chunk_size, "Message too big");

        let padlen = max_chunk_size * 128 - msg.len() as u64;
        if padlen > 0 {
            let mut padded_msg = Vec::new();
            padded_msg.extend_from_slice(msg);
            padded_msg.extend_from_slice(&vec![0u8; padlen as usize]);
            padded_msg
        } else {
            msg.to_vec()
        }
    }
}
