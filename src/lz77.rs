const BASE_SIZE: usize = 0x20000;
const NUM_PREVIOUS: usize = 4;
const MAX_LEN: usize = 0xFFFF;
const MAX_OFFSET: usize = 0xFFFF;

#[derive(Debug, Clone)]
pub struct Lz77Result {
    pub data: Vec<u32>,
    pub lengths: Vec<u32>,
    pub offsets: Vec<u32>,
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
    let mut data = Vec::with_capacity(nvals * 2);
    let mut lengths = Vec::with_capacity(nvals);
    let mut offsets = Vec::with_capacity(nvals);

    // Initialize the previous array (circular buffer for each possible value)
    let mut previous = vec![0i32; 0x20000 * (NUM_PREVIOUS + 3)];

    // Initialize circular buffers
    for i in 0..0x20000 {
        let base = (NUM_PREVIOUS + 3) * i;
        previous[base] = 0; // Number of items in circular buffer
        previous[base + 1] = 0; // Pointer to beginning of circular buffer
        previous[base + 2] = -2; // Last offset that had this value
    }

    let mut i = 0;
    while i < nvals {
        let first_offset = i.saturating_sub(MAX_OFFSET);

        if i != 0 {
            let mut largest_len = 0;
            let mut largest_offset = 0;

            // Search through circular buffer for possible string starts
            let val_idx = (NUM_PREVIOUS + 3) * (vals[i] as usize);
            let ncirc = previous[val_idx] as usize;

            for icirc in 0..ncirc {
                let iptr = if previous[val_idx + 1] - (icirc as i32) - 1 < 0 {
                    (previous[val_idx + 1] - (icirc as i32) - 1 + NUM_PREVIOUS as i32) as usize
                } else {
                    (previous[val_idx + 1] - (icirc as i32) - 1) as usize
                };

                let mut j = previous[val_idx + 3 + iptr] as usize;

                if j < first_offset {
                    break;
                }

                // Find matching sequences
                while j < i && vals[j] == vals[i] {
                    if j >= first_offset {
                        let mut k = 0;
                        while i + k < nvals && vals[j + k] == vals[i + k] {
                            k += 1;
                        }

                        if k > largest_len && (k >= (i - j) + 16 || (k > 4 && i - j == 1)) {
                            largest_len = k;
                            largest_offset = j;
                        }
                    }
                    j += 1;
                }
            }

            // Limit length to maximum
            if largest_len > MAX_LEN {
                largest_len = MAX_LEN;
            }

            if largest_len > 0 {
                // Encode the match
                if i - largest_offset == 1 {
                    data.push(0);
                } else {
                    data.push(1);
                    offsets.push((i - largest_offset) as u32);
                }
                lengths.push(largest_len as u32);

                // Add matched values to circular buffer
                for k in 0..largest_len {
                    add_circular(&mut previous, vals[i + k], (i + k) as i32);
                }
                i += largest_len - 1;
            } else {
                // No match found, store literal
                data.push(vals[i] + 2);
                add_circular(&mut previous, vals[i], i as i32);
            }
        } else {
            // First element is always literal
            data.push(vals[i] + 2);
            add_circular(&mut previous, vals[i], i as i32);
        }
        i += 1;
    }

    Lz77Result {
        data,
        lengths,
        offsets,
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
