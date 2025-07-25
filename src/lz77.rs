const BASE_SIZE: usize = 0x20000;
const NUM_PREVIOUS: usize = 4;
const MAX_LEN: usize = 0xFFFF;
const MAX_OFFSET: usize = 0xFFFF;

#[derive(Debug, Clone)]
pub struct Lz77Result {
    pub data: Vec<usize>,
    pub lengths: Vec<usize>,
    pub offsets: Vec<usize>,
}

fn add_circular(previous: &mut [i32], value: u32, position: i32) {
    let idx = (NUM_PREVIOUS + 3) * (value as usize);
    let ncirc = previous[idx] as usize;

    if ncirc < NUM_PREVIOUS {
        let ptr_idx = previous[idx + 1] as usize;
        previous[idx + 3 + ptr_idx] = position;
        previous[idx] += 1;
        previous[idx + 1] = ((ptr_idx + 1) % NUM_PREVIOUS) as i32;
    } else {
        let ptr_idx = previous[idx + 1] as usize;
        previous[idx + 3 + ptr_idx] = position;
        previous[idx + 1] = ((ptr_idx + 1) % NUM_PREVIOUS) as i32;
    }
}

pub fn ptngc_comp_to_lz77(vals: &[u32]) -> Lz77Result {
    let nvals = vals.len();
    let mut noff = 0;
    let mut ndat = 0;
    let mut nlen = 0;

    let mut data = vec![0; nvals];
    let mut lengths = vec![0; nvals];
    let mut offsets = vec![0; nvals];

    // Initialize the previous array (circular buffer for each possible value)
    let mut previous = vec![0i32; 0x20000 * (NUM_PREVIOUS + 3)];

    // Initialize circular buffers
    for chunk in previous.chunks_mut(NUM_PREVIOUS + 3) {
        // chunk[0] = 0; // Number of items in circular buffer
        // chunk[1] = 0; // Pointer to beginning of circular buffer
        chunk[2] = -2; // Last offset that had this value
    }

    let mut j;
    let mut i = 0;
    while i < nvals {
        let firstoffset = i.saturating_sub(MAX_OFFSET);

        if i != 0 {
            let mut largest_len = 0;
            let mut largest_offset = 0;
            // Is this identical to a previous offset? Prefer close
            // values for offset. Search through circular buffer for the
            // possible values for the start of this string
            let v = usize::try_from(vals[i]).expect("usize from u32");
            let ncirc = previous[(NUM_PREVIOUS + 3) * v];
            for icirc in 0..ncirc {
                let mut iptr = previous[(NUM_PREVIOUS + 3) * v + 1] - icirc - 1;
                if iptr < 0 {
                    iptr += i32::try_from(NUM_PREVIOUS).expect("i32 from usize");
                }
                j = previous
                    [(NUM_PREVIOUS + 3) * v + 3 + usize::try_from(iptr).expect("usize from i32")];
                if j < i32::try_from(firstoffset).expect("i32 from usize") {
                    break;
                }

                let mut j_usize = usize::try_from(j).expect("usize from i32");
                while (j_usize < i)
                    && (usize::try_from(vals[j_usize]).expect("usize from u32") == v)
                {
                    if j_usize >= firstoffset {
                        let mut k = 0;
                        while (k + i) < nvals {
                            if vals[j_usize + k] != vals[i + k] {
                                break;
                            }
                            k += 1;
                        }

                        if (k > largest_len)
                            && ((k >= (i - j_usize) + 16) || ((k > 4) && (i - j_usize == 1)))
                        {
                            largest_len = k;
                            largest_offset = j_usize;
                        }
                    }
                    j_usize += 1;
                    j += 1;
                }
            }

            // Check how to write this info
            if largest_len > MAX_LEN {
                largest_len = MAX_LEN;
            }

            if largest_len > 0 {
                if i - largest_offset == 1 {
                    data[ndat] = 0;
                    ndat += 1;
                } else {
                    data[ndat] = 1;
                    ndat += 1;
                    offsets[noff] = i - largest_offset;
                    noff += 1;
                }
                lengths[nlen] = largest_len;
                nlen += 1;

                // Add these values to the circular buffer
                for k in 0..largest_len {
                    add_circular(
                        &mut previous,
                        vals[i + k],
                        i32::try_from(i + k).expect("i32 from usize"),
                    );
                }
                i += largest_len - 1;
            } else {
                data[ndat] = v + 2;
                ndat += 1;
                // Add this value to circular buffer
                add_circular(
                    &mut previous,
                    u32::try_from(v).expect("u32 from usize"),
                    i32::try_from(i).expect("i32 from usize"),
                );
            }
        } else {
            data[ndat] = usize::try_from(vals[i] + 2).expect("usize from u32");
            ndat += 1;
            // Add this value to circular buffer
            add_circular(
                &mut previous,
                vals[i],
                i32::try_from(i).expect("i32 from usize"),
            );
        }
        i += 1;
    }

    Lz77Result {
        data: data[..ndat].to_vec(),
        lengths: lengths[..nlen].to_vec(),
        offsets: offsets[..noff].to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let input: &[u32] = &[];
        let result = ptngc_comp_to_lz77(input);

        assert_eq!(result.data, vec![]);
        assert_eq!(result.lengths, vec![]);
        assert_eq!(result.offsets, vec![]);
    }

    #[test]
    fn test_single_element() {
        let input = &[42];
        let result = ptngc_comp_to_lz77(input);

        assert_eq!(result.data, vec![44]); // 42 + 2
        assert_eq!(result.lengths, vec![]);
        assert_eq!(result.offsets, vec![]);
    }

    #[test]
    fn test_no_repetition() {
        let input = &[1, 2, 3, 4, 5];
        let result = ptngc_comp_to_lz77(input);

        assert_eq!(result.data, vec![3, 4, 5, 6, 7]); // each + 2
        assert_eq!(result.lengths, vec![]);
        assert_eq!(result.offsets, vec![]);
    }

    #[test]
    fn test_simple_repetition() {
        let input = &[1, 2, 1, 2];
        let result = ptngc_comp_to_lz77(input);

        assert_eq!(result.data, vec![3, 4, 3, 4]);
        assert_eq!(result.lengths, vec![]);
        assert_eq!(result.offsets, vec![]);
    }

    #[test]
    fn test_adjacent_repetition() {
        let input = &[5, 5, 5, 5, 5];
        let result = ptngc_comp_to_lz77(input);

        assert_eq!(result.data, vec![7, 7, 7, 7, 7]);
        assert_eq!(result.lengths, vec![]);
        assert_eq!(result.offsets, vec![]);
    }

    #[test]
    fn test_complex_pattern() {
        let input = &[1, 2, 3, 1, 2, 3, 4, 1, 2, 3];
        let result = ptngc_comp_to_lz77(input);

        assert_eq!(result.data, vec![3, 4, 5, 3, 4, 5, 6, 3, 4, 5]);
        assert_eq!(result.lengths, vec![]);
        assert_eq!(result.offsets, vec![]);
    }

    #[test]
    fn test_large_values_near_max() {
        let input = &[0xFFFF, 0xFFFE, 0xFFFF, 0xFFFE];
        let result = ptngc_comp_to_lz77(input);

        assert_eq!(result.data, vec![65537, 65536, 65537, 65536]);
        assert_eq!(result.lengths, vec![]);
        assert_eq!(result.offsets, vec![]);
    }

    #[test]
    fn test_overlapping_matches() {
        let input = &[1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6];
        let result = ptngc_comp_to_lz77(input);

        assert_eq!(result.data, vec![3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8]);
        assert_eq!(result.lengths, vec![]);
        assert_eq!(result.offsets, vec![]);
    }

    #[test]
    fn test_long_repetitive_sequence() {
        let input: Vec<u32> = (0..100).map(|i| (i % 10) as u32).collect();
        let result = ptngc_comp_to_lz77(&input);

        assert_eq!(result.data, vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1]);
        assert_eq!(result.lengths, vec![90]);
        assert_eq!(result.offsets, vec![10]);
    }
}
