pub(crate) fn ptngc_comp_conv_to_mtf_partial3(vals: &[u32], nvals: usize, valsmtf: &mut [u8]) {
    let mut tmp = vec![0u8; nvals];

    for j in 0..3 {
        for i in 0..nvals {
            tmp[i] = ((vals[i] >> (8 * j)) & 0xFF) as u8;
        }
        comp_conv_to_mtf_byte(&tmp, &mut valsmtf[j * nvals..]);
    }
}

/// Move to front coding
/// Acceptable inputs are max 24 bits (0-0xFFFFFF)
///
/// # Panics
/// - If `vals.len() != valsmtf.len()`.
/// - If any `v` is not found in `dict`.
pub(crate) fn ptngc_comp_conv_to_mtf(vals: &[u32], dict: &[u32], valsmtf: &mut [u32]) {
    assert_eq!(vals.len(), valsmtf.len(), "input/output length mismatch");

    // Initialize the dynamic symbol list from the provided dict
    let mut list: Vec<u32> = dict.to_vec();

    for (i, &v) in vals.iter().enumerate() {
        // Find the index of `v` in the list
        let pos = list
            .iter()
            .position(|&sym| sym == v)
            .expect("input symbol not found in dictionary");

        // Emit that index as the MTF code
        valsmtf[i] = pos as u32;

        // Move this symbol to front, if not already there
        if pos != 0 {
            // remove at pos and re‑insert at front
            let sym = list.remove(pos);
            list.insert(0, sym);
        }
    }
}

/// "Partial" MTF. Byte based. Move to front coding.
/// Acceptable inputs are max 8 bits (0-0xFF)
pub(crate) fn comp_conv_to_mtf_byte(vals: &[u8], valsmtf: &mut [u8]) {
    let mut list = (1..256 + 1).collect::<Vec<i32>>();
    let dict = (0..256).collect::<Vec<i32>>();

    list[255] = -1;
    let mut head = 0;

    for i in 0..vals.len() {
        let v = i32::from(vals[i]);

        // Find how early in the dict the value is
        let mut ptr = head;
        let mut oldptr = -1;
        let mut r = 0;
        while dict[usize::try_from(ptr).expect("usize from i32")] != v {
            oldptr = ptr;
            ptr = list[usize::try_from(ptr).expect("usize from i32")];
            r += 1;
        }
        valsmtf[i] = r as u8;

        // Move it to fron in list
        // Is it the head? Then it is already at the front
        if oldptr != -1 {
            // Remove it from inside the list
            list[usize::try_from(oldptr).expect("usize from i32")] =
                list[usize::try_from(ptr).expect("usize from i32")];

            // Move it to the front
            list[usize::try_from(ptr).expect("usize from i32")] = head;
            head = ptr;
        }
    }
}

/// Decode a byte-based Move-To-Front encoding.
///
/// # Panics
///
/// Panics if `valsmtf.len() != vals.len()`.
pub(crate) fn comp_conv_from_mtf_byte(valsmtf: &[u8], vals: &mut [u8]) {
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

pub(crate) fn ptngc_comp_conv_to_mtf_partial(vals: &[u32], valsmtf: &mut [u32]) {
    let nvals = vals.len();
    assert_eq!(valsmtf.len(), nvals, "output length must match input");

    // tmp will hold [0..n) = input bytes, [n..2n) = MTF codes
    let mut tmp = vec![0u8; nvals * 2];

    // zero the output
    valsmtf.fill(0);

    for byte_shift in 0..3 {
        // extract the `byte_shift`th byte of each word
        for i in 0..nvals {
            tmp[i] = ((vals[i] >> (8 * byte_shift)) & 0xFF) as u8;
        }
        // run byte‑based MTF on that byte‑slice, writing codes into tmp[n..2n]
        let (src, dst) = tmp.split_at_mut(nvals);
        comp_conv_to_mtf_byte(src, dst);

        // pack each code byte back into the corresponding u32 output
        for i in 0..nvals {
            valsmtf[i] |= (tmp[nvals + i] as u32) << (8 * byte_shift);
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
