use std::cmp::min;
use std::fs::File;
use std::io::{Read, Write};

use crate::MAX_STR_LEN;

/// Represents the endianness of 32 bit values
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Endianness32 {
    Little,
    Big,
    BytePairSwap,
}

/// Represents the endianness of 32 bit values
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Endianness64 {
    Little,
    Big,
    QuadSwap,
    BytePairSwap,
    ByteSwap,
}

pub(crate) type SwapFn32 = fn(Endianness32, &mut u32);
pub(crate) type SwapFn64 = fn(Endianness64, &mut u64);

/// Swaps the byte order of a 32-bit numerical variable to big endian
pub fn swap_byte_order_big_endian_32(endianness: Endianness32, raw: &mut u32) {
    match endianness {
        // Byte order is reversed
        Endianness32::Little => {
            *raw = raw.swap_bytes();
        }
        // Already correct
        Endianness32::Big => {}
        // Byte pair swap
        Endianness32::BytePairSwap => {
            *raw = ((*raw & 0xFFFF_0000) >> 16) | ((*raw & 0x0000_FFFF) << 16);
        }
    }
}

/// Swaps the byte order of a 64-bit numerical variable to big endian
pub fn swap_byte_order_big_endian_64(endianness: Endianness64, raw: &mut u64) {
    match endianness {
        // Byte order is reversed
        Endianness64::Little => {
            *raw = raw.swap_bytes();
        }
        // Already correct
        Endianness64::Big => {}
        Endianness64::QuadSwap => {
            *raw = ((*raw & 0xFFFF_FFFF_0000_0000) >> 32) | ((*raw & 0x0000_0000_FFFF_FFFF) << 32);
        }
        Endianness64::BytePairSwap => {
            *raw = ((*raw & 0xFFFF_0000_FFFF_0000) >> 16) | ((*raw & 0x0000_FFFF_0000_FFFF) << 16);
        }
        Endianness64::ByteSwap => {
            *raw = ((*raw & 0xFF00_FF00_FF00_FF00) >> 8) | ((*raw & 0x00FF_00FF_00FF_00FF) << 8);
        }
    }
}

/// Swaps the byte order of a 32-bit numerical variable to little endian
pub fn swap_byte_order_little_endian_32(endianness: Endianness32, raw: &mut u32) {
    match endianness {
        Endianness32::Little => {
            // Already little‐endian; no action needed
        }
        Endianness32::BytePairSwap => {
            // Swap each byte pair: 0xAABBCCDD → 0xBBAADDCC
            *raw = ((*raw & 0xFF00_FF00) >> 8) | ((*raw & 0x00FF_00FF) << 8);
        }
        Endianness32::Big => {
            // Reverse all bytes: 0xAABBCCDD → 0xDDCCBBAA
            *raw = raw.swap_bytes();
        }
    }
}

/// Swaps the byte order of a 64-bit numerical variable to little endian
pub fn swap_byte_order_little_endian_64(endianness: Endianness64, raw: &mut u64) {
    match endianness {
        // Already correct
        Endianness64::Little => {}
        // Byte order is reversed
        Endianness64::Big => *raw = raw.swap_bytes(),
        // Byte quad swapped big endian to little endian
        Endianness64::QuadSwap => {
            *raw = ((*raw & 0xFF00_0000_FF00_0000) >> 24)
                | ((*raw & 0x00FF_0000_00FF_0000) >> 8)
                | ((*raw & 0x0000_FF00_0000_FF00) << 8)
                | ((*raw & 0x0000_00FF_0000_00FF) << 24);
        }
        // Byte pair swapped big endian to little endian
        Endianness64::BytePairSwap => {
            *raw = ((*raw & 0xFF00_FF00_0000_0000) >> 40)
                | ((*raw & 0x00FF_00FF_0000_0000) >> 24)
                | ((*raw & 0x0000_0000_FF00_FF00) << 24)
                | ((*raw & 0x0000_0000_00FF_00FF) << 40);
        }
        // Byte pair swapped big endian to little endian
        Endianness64::ByteSwap => {
            *raw = ((*raw & 0xFFFF_0000_0000_0000) >> 48)
                | ((*raw & 0x0000_FFFF_0000_0000) >> 16)
                | ((*raw & 0x0000_0000_FFFF_0000) << 16)
                | ((*raw & 0x0000_0000_0000_FFFF) << 48);
        }
    }
}

/// Read exactly N bytes from `reader` into a `[u8; N]`.  
///
/// # Panic
/// Panics (or unwraps) on any I/O error.
pub(crate) fn read_exact_array<const N: usize, R: Read>(reader: &mut R) -> [u8; N] {
    let mut buf = [0u8; N];
    reader
        .read_exact(&mut buf)
        .expect("could not read bytes from file");
    buf
}

pub fn read_u8(input_file: &mut File) -> u8 {
    read_exact_array::<1, _>(input_file)[0]
}

pub fn read_u64(
    input_file: &mut File,
    endianness: Endianness64,
    input_swap64: Option<SwapFn64>,
) -> u64 {
    let raw_bytes = read_exact_array(input_file);
    let mut val: u64 = u64::from_ne_bytes(raw_bytes);
    if let Some(swap_fn_64) = input_swap64 {
        swap_fn_64(endianness, &mut val);
    }
    val
}

pub fn read_i64(
    input_file: &mut File,
    endianness: Endianness64,
    input_swap64: Option<SwapFn64>,
) -> i64 {
    let intermediate_val: u64 = read_u64(input_file, endianness, input_swap64);
    // Reinterpret the bit‐pattern as an i64:
    i64::from_ne_bytes(intermediate_val.to_ne_bytes())
}

pub fn read_f64(
    input_file: &mut File,
    endianness: Endianness64,
    input_swap64: Option<SwapFn64>,
) -> f64 {
    let u: u64 = read_u64(input_file, endianness, input_swap64);
    f64::from_bits(u)
}

pub fn read_u32_and_swap(
    input_file: &mut File,
    endianness: Endianness32,
    input_swap32: Option<SwapFn32>,
) -> u32 {
    let raw_bytes: [u8; 4] = read_exact_array(input_file);
    let mut val: u32 = u32::from_ne_bytes(raw_bytes);
    if let Some(swap_fn_32) = input_swap32 {
        swap_fn_32(endianness, &mut val);
    }
    val
}

pub fn read_i32_and_swap(
    input_file: &mut File,
    endianness: Endianness32,
    input_swap32: Option<SwapFn32>,
) -> i32 {
    let u: u32 = read_u32_and_swap(input_file, endianness, input_swap32);
    i32::from_ne_bytes(u.to_ne_bytes())
}

pub fn read_f32_and_swap(
    input_file: &mut File,
    endianness: Endianness32,
    input_swap32: Option<SwapFn32>,
) -> f32 {
    let u: u32 = read_u32_and_swap(input_file, endianness, input_swap32);
    f32::from_bits(u)
}

pub fn read_bool_le_bytes(input_file: &mut File) -> bool {
    let buf = read_exact_array::<1, _>(input_file);
    buf[0] != 0
}

pub(crate) fn fwrite_str<W: Write>(output_file: &mut W, str_data: &str) {
    let mut bytes = str_data.as_bytes().to_vec();
    bytes.push(0); // null-terminate
    let len = bytes.len().min(MAX_STR_LEN);
    output_file.write_all(&bytes[..len]);

    // TODO: HASH
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

pub fn write_u64(
    output_file: &mut File,
    src: u64,
    endianness: Endianness64,
    output_swap64: Option<SwapFn64>,
) {
    let mut temp_u64 = src;
    if let Some(swap_fn_64) = output_swap64 {
        swap_fn_64(endianness, &mut temp_u64);
    }
    let out_bytes = temp_u64.to_ne_bytes();
    output_file
        .write_all(&out_bytes)
        .expect("to be able to write to output_file");
}

pub fn write_i64(
    output_file: &mut File,
    src: i64,
    endianness: Endianness64,
    output_swap64: Option<SwapFn64>,
) {
    // Convert i64 → [u8; 8] in *native* endianness:
    let src_bytes: [u8; 8] = src.to_ne_bytes();

    // Reinterpret those 8 bytes as a u64 (bitwise identical):
    let mut bits: u64 = u64::from_ne_bytes(src_bytes);

    if let Some(swap_fn_64) = output_swap64 {
        swap_fn_64(endianness, &mut bits);
    }
    let out_bytes = bits.to_ne_bytes();
    output_file
        .write_all(&out_bytes)
        .expect("to be able to write to output_file");
}

pub fn write_f64(
    output_file: &mut File,
    src: f64,
    endianness: Endianness64,
    output_swap64: Option<SwapFn64>,
) {
    // Convert f64 → [u8; 8] in *native* endianness:
    let src_bytes: [u8; 8] = src.to_ne_bytes();

    // Reinterpret those 8 bytes as a u64 (bitwise identical):
    let mut bits: u64 = u64::from_ne_bytes(src_bytes);

    if let Some(swap_fn_64) = output_swap64 {
        swap_fn_64(endianness, &mut bits);
    }
    let out_bytes = bits.to_ne_bytes();
    output_file
        .write_all(&out_bytes)
        .expect("to be able to write to output_file");
}

pub fn write_u32(
    output_file: &mut File,
    src: u32,
    endianness: Endianness32,
    output_swap32: Option<SwapFn32>,
) {
    let mut temp_u32 = src;
    if let Some(swap_fn_32) = output_swap32 {
        swap_fn_32(endianness, &mut temp_u32);
    }
    let out_bytes = temp_u32.to_ne_bytes();
    output_file
        .write_all(&out_bytes)
        .expect("to be able to write to output_file");
}

pub fn write_i32(
    output_file: &mut File,
    src: i32,
    endianness: Endianness32,
    output_swap32: Option<SwapFn32>,
) {
    // Convert i32 → [u8; 4] in *native* endianness:
    let src_bytes: [u8; 4] = src.to_ne_bytes();

    // Reinterpret those 4 bytes as a u32 (bitwise identical):
    let mut bits: u32 = u32::from_ne_bytes(src_bytes);

    if let Some(swap_fn_32) = output_swap32 {
        swap_fn_32(endianness, &mut bits);
    }
    let out_bytes = bits.to_ne_bytes();
    output_file
        .write_all(&out_bytes)
        .expect("to be able to write to output_file");
}

pub fn write_f32(
    output_file: &mut File,
    src: f32,
    endianness: Endianness32,
    output_swap32: Option<SwapFn32>,
) {
    // Convert f32 → [u8; 4] in *native* endianness:
    let src_bytes: [u8; 4] = src.to_ne_bytes();

    // Reinterpret those 4 bytes as a u32 (bitwise identical):
    let mut bits: u32 = u32::from_ne_bytes(src_bytes);

    if let Some(swap_fn_32) = output_swap32 {
        swap_fn_32(endianness, &mut bits);
    }
    let out_bytes = bits.to_ne_bytes();
    output_file
        .write_all(&out_bytes)
        .expect("to be able to write to output_file");
}

pub fn write_bool(output_file: &mut File, value: bool) {
    let byte = if value { 1u8 } else { 0u8 };
    output_file
        .write_all(&[byte])
        .expect("to be able to write bool to output_file");
    // TODO: HASH
}

pub(crate) fn bounded_len(s: &str) -> usize {
    min(s.len().checked_add(1).unwrap_or(MAX_STR_LEN), MAX_STR_LEN)
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
