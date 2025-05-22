use std::fs::File;
use std::io::Read;

use crate::MAX_STR_LEN;

/// Read exactly N bytes from `reader` into a `[u8; N]`.  
/// Panics (or unwraps) on any I/O error.
fn read_exact_array<const N: usize, R: Read>(reader: &mut R) -> [u8; N] {
    let mut buf = [0u8; N];
    reader.read_exact(&mut buf).unwrap();
    buf
}

pub fn read_i64_le_bytes(input_file: &mut File) -> i64 {
    i64::from_le_bytes(read_exact_array::<8, _>(input_file))
}

pub fn read_u64_le_bytes(input_file: &mut File) -> u64 {
    u64::from_le_bytes(read_exact_array::<8, _>(input_file))
}

pub fn read_bool_le_bytes(input_file: &mut File) -> bool {
    let buf = read_exact_array::<1, _>(input_file);
    buf[0] != 0
}

pub fn fread_str<R: Read>(input_file: &mut R) -> String {
    // Accumulate bytes here, including the trailing 0:
    let mut buf: Vec<u8> = Vec::new();

    // Temporary single-byte buffer for reading:
    let mut byte = [0u8; 1];

    loop {
        // Try to read exactly one byte
        match input_file.read_exact(&mut byte) {
            Ok(()) => {
                let b = byte[0];
                buf.push(b);

                // // If hash_mode, feed it into the MD5 state right away
                // if hash_mode {
                //     md5_state.update(&[b]);
                // }

                // Stop on NUL or if we exceed max length
                if b == 0 {
                    break;
                }
                if buf.len() >= MAX_STR_LEN {
                    // We hit the same “limit” as the C version (avoid overflow).
                    // In C, it would stop copying; here we break and use what we have.
                    break;
                }
            }
            Err(_) => {
                panic!("something went wrong when reading string");
                // // EOF or other I/O error. Distinguish EOF (Minor) vs. other errors (Critical).
                // if e.kind() == io::ErrorKind::UnexpectedEof {
                //     // Clear the EOF flag (if needed) and return “minor” failure.
                //     return Err(ReadStrError::Minor);
                // } else {
                //     return Err(ReadStrError::Critical(e));
                // }
            }
        }
    }

    // If we read exactly one byte and it was NUL, that's effectively an empty string.
    // The C version would then allocate a length-1 buffer and store "\0".
    // We’ll treat that as an empty Rust String here (dropping the trailing zero):
    if buf.len() == 1 && buf[0] == 0 {
        return String::new();
    }

    // Otherwise, we have some bytes like [b'a', b'b', b'c', 0]. Drop the final 0:
    if let Some(&0) = buf.last() {
        buf.pop();
    }

    // Convert to UTF-8, with a lossy fallback if it wasn’t valid UTF-8:
    match String::from_utf8(buf) {
        Ok(valid) => valid,
        Err(e) => String::from_utf8_lossy(&e.into_bytes()).into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper that wraps a byte slice in a Cursor to mimic a File.
    fn make_cursor(data: &[u8]) -> Cursor<Vec<u8>> {
        Cursor::new(data.to_vec())
    }

    #[test]
    fn test_simple_string_with_null() {
        // "hello\0world" → should read "hello" and stop at the first NUL.
        let mut cursor = make_cursor(b"hello\0world");
        let result = fread_str(&mut cursor);
        assert_eq!(result, "hello");
        // After reading, the cursor's position should be right after the NUL.
        assert_eq!(cursor.position() as usize, "hello\0".len());
    }

    #[test]
    fn test_empty_string_only_null() {
        // "\0" → should return an empty string.
        let mut cursor = make_cursor(b"\0");
        let result = fread_str(&mut cursor);
        assert_eq!(result, "");
        assert_eq!(cursor.position() as usize, 1);
    }

    #[test]
    fn test_truncate_at_max_length() {
        // Create a buffer of exactly MAX_STR_LEN bytes, none of which are 0.
        let long_data: Vec<u8> = vec![b'x'; MAX_STR_LEN];
        let mut cursor = make_cursor(&long_data);
        let result = fread_str(&mut cursor);
        // Since there's no NUL, fread_str should read exactly MAX_STR_LEN bytes then stop.
        assert_eq!(result.len(), MAX_STR_LEN);
        assert!(result.chars().all(|c| c == 'x'));
        assert_eq!(cursor.position() as usize, MAX_STR_LEN);
    }

    #[test]
    #[should_panic(expected = "something went wrong when reading string")]
    fn test_unexpected_eof_panics() {
        // Provide fewer bytes than MAX_STR_LEN and no NUL: e.g. "abc" only.
        // After reading 'a','b','c', the next read_exact will hit EOF and panic.
        let mut cursor = make_cursor(b"abc");
        let _ = fread_str(&mut cursor);
    }

    #[test]
    fn test_invalid_utf8_lossy() {
        // Bytes [0xFF, 0xFE, 0] → after trimming NUL, buf = [0xFF, 0xFE].
        // from_utf8() should fail; from_utf8_lossy() yields "��".
        let data = [0xFF, 0xFE, 0];
        let mut cursor = make_cursor(&data);
        let result = fread_str(&mut cursor);
        assert_eq!(result, "��");
    }
}
