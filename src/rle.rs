/// Run length encoding
/// acceptable inputs are about 16 bits (0-0xFFFF)
/// If input is 0-N output will be values of 0-(N+2)
pub(crate) fn ptngc_comp_conv_to_rle(vals: &[u32], min_rle: usize) -> Vec<u32> {
    if vals.is_empty() {
        return Vec::new();
    }

    let mut rle = Vec::with_capacity(vals.len());
    let mut current_val = vals[0] as i32;
    let mut run_length = 1;

    // Process all values after the first
    for &val in &vals[1..] {
        let val_i32 = val as i32;
        if val_i32 == current_val {
            run_length += 1;
        } else {
            // Encode the current run
            add_rle(&mut rle, current_val, run_length, min_rle);
            current_val = val_i32;
            run_length = 1;
        }
    }

    // Don't forget the last run
    add_rle(&mut rle, current_val, run_length, min_rle);

    rle
}

pub(crate) fn ptngc_comp_conv_from_rle(rle: &[u32], output_len: usize) -> Vec<u32> {
    let mut vals = Vec::with_capacity(output_len);
    let mut i = 0;
    let mut j = 0;

    while i < output_len {
        let mut len = 0;
        let mut mask = 0x1;
        let mut v = rle[j];
        j += 1;
        let mut has_rle = false;

        while v < 2 {
            if v != 0 {
                len |= mask;
            }
            mask <<= 1;
            has_rle = true;
            v = rle[j];
            j += 1;
        }

        if !has_rle {
            len = 1;
        } else {
            len |= mask;
        }

        for _ in 0..len {
            vals.push(v.saturating_sub(2));
            i += 1;
        }
    }

    vals
}

fn add_rle(rle: &mut Vec<u32>, value: i32, count: usize, min_rle: usize) {
    let mut remaining_count = count;

    if count > min_rle {
        // Encode run length in binary
        let mut run = count as u32;
        while run > 1 {
            rle.push(run & 1);
            run >>= 1;
        }
        remaining_count = 1;
    }

    let encoded_value = (value + 2) as u32;
    for _ in 0..remaining_count {
        rle.push(encoded_value);
    }
}

#[cfg(test)]
mod tests {
    use super::{ptngc_comp_conv_from_rle, ptngc_comp_conv_to_rle};

    #[test]
    fn it_works() {
        let vals = vec![1, 2, 3];
        let min_rle = 3;
        let encoded = ptngc_comp_conv_to_rle(&vals, min_rle);

        assert_eq!(encoded, vec![3, 4, 5]);
    }

    #[test]
    fn repetitive() {
        let vals = vec![7, 7, 7, 7, 7];
        let min_rle = 3;
        let encoded = ptngc_comp_conv_to_rle(&vals, min_rle);

        assert_eq!(encoded, vec![1, 0, 9]);
    }

    #[test]
    fn exact_min_rle() {
        let vals = vec![9, 9, 9, 9];
        let min_rle = 4;
        let encoded = ptngc_comp_conv_to_rle(&vals, min_rle);

        assert_eq!(encoded, vec![11, 11, 11, 11]);
    }

    #[test]
    fn empty_input() {
        let vals = vec![];
        let min_rle = 0;
        let encoded = ptngc_comp_conv_to_rle(&vals, min_rle);

        assert_eq!(encoded, vec![]);
    }

    #[test]
    fn roundtrip() {
        let vals = vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6];
        let min_rle = 2;
        let encoded = ptngc_comp_conv_to_rle(&vals, min_rle);
        let decoded = ptngc_comp_conv_from_rle(&encoded, vals.len());

        assert_eq!(vals, decoded);
    }
}
