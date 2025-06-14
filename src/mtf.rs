pub(crate) fn ptngc_comp_conv_to_mtf_partial3(vals: &[u32], nvals: usize, valsmtf: &mut [u8]) {
    let mut tmp = vec![0u8; nvals];

    for j in 0..3 {
        for i in 0..nvals {
            tmp[i] = ((vals[i] >> (8 * j)) & 0xFF) as u8;
        }
        comp_conv_to_mtf_byte(&tmp, &mut valsmtf[j * nvals..]);
    }
}

/// "Partial" MTF. Byte based. Move to front coding.
/// Acceptable inputs are max 8 bits (0-0xFF)
///
/// # Panics
///
/// Panics if `vals.len() != valsmtf.len()`.
fn comp_conv_to_mtf_byte(vals: &[u8], valsmtf: &mut [u8]) {
    assert_eq!(vals.len(), valsmtf.len(), "input/output length mismatch");

    // Initialize the symbol list to [0,1,2,…,255].
    let mut list: Vec<u8> = (0u8..=255).collect();

    for (i, &v) in vals.iter().enumerate() {
        // Find the index (r) of `v` in the list
        let pos = list
            .iter()
            .position(|&sym| sym == v)
            .expect("input byte out of 0..=255 range");

        // Emit that index as the MTF code
        valsmtf[i] = pos as u8;

        // Move this symbol to front, if not already there
        if pos != 0 {
            list.remove(pos);
            list.insert(0, v);
        }
    }
}

/// Decode a byte-based Move-To-Front encoding.
///
/// # Panics
///
/// Panics if `valsmtf.len() != vals.len()`.
pub fn comp_conv_from_mtf_byte(valsmtf: &[u8], vals: &mut [u8]) {
    assert_eq!(valsmtf.len(), vals.len(), "input/output length mismatch");

    // Initialize the symbol list 0,1,2,…,255
    let mut list: Vec<u8> = (0u8..=255).collect();

    for (i, &r) in valsmtf.iter().enumerate() {
        let pos = r as usize;
        // Take the symbol at position `pos`
        let symbol = list[pos];
        vals[i] = symbol;
        // Move that symbol to front, if not already there
        if pos != 0 {
            list.remove(pos);
            list.insert(0, symbol);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(input: &[u8]) {
        let mut encoded = vec![0u8; input.len()];
        comp_conv_to_mtf_byte(input, &mut encoded);

        let mut decoded = vec![0u8; input.len()];
        comp_conv_from_mtf_byte(&encoded, &mut decoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rt_empty() {
        roundtrip(&[]);
    }
    #[test]
    fn rt_single() {
        roundtrip(&[99]);
    }
    #[test]
    fn rt_repeat() {
        roundtrip(&[7, 7, 7, 7, 7]);
    }
    #[test]
    fn rt_sequence() {
        roundtrip(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
    #[test]
    fn rt_banana() {
        roundtrip(b"banana");
    }
    #[test]
    fn rt_pattern() {
        roundtrip(&[5, 2, 5, 2, 5, 3, 5, 2]);
    }
}
