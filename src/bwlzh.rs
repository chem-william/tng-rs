use std::cmp::Ordering;

use log::debug;
const MAX_VALS_PER_BLOCK: usize = 200000;

// TODO: enable these as compile-time features?
const PARTIAL_MTF3: bool = true;
const PARTIAL_MTF: bool = false;

pub(crate) const fn bwlzh_get_buflen(nvals: usize) -> usize {
    132000 + nvals * 8 + 12 * ((nvals + MAX_VALS_PER_BLOCK) / MAX_VALS_PER_BLOCK)
}

pub(crate) const fn ptngc_comp_huff_buflen(nvals: i32) -> i32 {
    132000 + nvals * 8
}

fn bwlzh_compress_no_lz77() {
    todo!()
}

/// Compress the integers (positive, small integers are preferable) using bwlzh compression. The
/// unsigned char *output should be allocated to be able to hold worst case. You can obtain this
/// length conveniently by calling `comp_get_buflen()`
pub(crate) fn bwlzh_compress(vals: &[u32], nvals: i32, output: &mut [u8]) -> i32 {
    bwlzh_compress_gen(vals, nvals, output, 1, 0)
}

pub(crate) fn bwlzh_compress_gen(
    vals: &[u32],
    nvals: i32,
    output: &mut [u8],
    enable_lz77: i32,
    verbose: i32,
) -> i32 {
    let mut outdata = 0;
    let mut valsleft = 0;
    let mut valstart = 0;
    let mut thisvals: usize = 0;
    let bwlzhuff =
        vec![0; usize::try_from(ptngc_comp_huff_buflen(3 * nvals)).expect("usize from i32")];
    let total_len = MAX_VALS_PER_BLOCK * 18;
    let mut tmpmem = vec![0u32; total_len];

    let (vals16, rest) = tmpmem.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (bwt, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (mtf, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (rle, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (offsets, lens) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);

    // TODO: enable feature flag "partial-mtf3"
    let mtf3 = vec![0; MAX_VALS_PER_BLOCK * 3 * 3];

    debug!("Number of input values: {nvals}");

    // Store the number of real values in the whole block
    let bytes = (nvals as u32).to_le_bytes(); // [u8; 4]
    output[outdata..outdata + 4].copy_from_slice(&bytes);
    outdata += 4;

    valsleft = usize::try_from(nvals).expect("usize from i32");
    valstart = 0;

    while valsleft > 0 {
        let mut reducealgo = 1; // Reduce algo is LZ77
        if enable_lz77 == 0 {
            reducealgo = 0;
        }
        thisvals = valsleft;
        if thisvals > MAX_VALS_PER_BLOCK {
            thisvals = MAX_VALS_PER_BLOCK;
        }
        valsleft -= thisvals;
        debug!("Creating vals16 block from {thisvals} values");
    }

    let nvals16 = ptngc_comp_conv_to_vals16(&vals[valstart..], vals16);
    valstart += thisvals;

    debug!("Resulting vals16 values: {nvals16}");
    debug!("BWT");
    let bwt_index = ptngc_comp_to_bwt(vals16, nvals16);

    0
}

/// Burrows-Wheeler transform
fn ptngc_comp_to_bwt(vals: &[u32], nvals: usize) -> (i32, i32) {
    if nvals > 0xFFFFFF {
        println!("BWT cannot pack more than {} values.", 0xFFFFFF);
    }

    // Also note that repeat pattern k (kmax) cannot be larger than 255
    let mut indices: Vec<usize> = (0..nvals).collect();

    // Find the length of the initial repeating pattern for the strings.
    // First mark that the index does not have a found repeating string.
    let mut nrepeat = vec![0; nvals];

    for i in 0..nvals {
        // If we have not already found a repeating string we must find it
        if nrepeat[i] == 0 {
            let maxrepeat = nvals * 2;
            let mut good_j = -1;
            let mut good_k = 0;
            let kmax = 16;
            // Track repeating patterns.
            // k=1 corresponds to AAAAA...
            // k=2 corresponds to ABABAB...
            // k=3 corresponds to ABCABCABCABC...
            // k=4 corresponds to ABCDABCDABCD...
            // etc.
            let mut k = kmax;
            'k_loop: while k >= 1 {
                debug!("Trying k={} at i={}", k, i);

                // for j = k; j < maxrepeat; j += k
                let mut j = k;
                while j < maxrepeat {
                    debug!("Trying j={} at i={} for k={}", j, i, k);

                    // check if vals[i+m] == vals[i+j+m] for m in 0..k
                    let mut is_equal = true;
                    let mut m = 0;

                    while m < k {
                        let a = &vals[(i + m) % nvals];
                        let b = &vals[(i + j + m) % nvals];
                        if a != b {
                            is_equal = false;
                            break;
                        }
                        m += 1;
                    }

                    if is_equal {
                        // we have a repeat of length `k` at offset `j`
                        let new_j = i32::try_from(if j + k > maxrepeat { j } else { j + k })
                            .expect("i32 from usize");
                        if new_j > good_j || (new_j == good_j && k < good_k) {
                            // We have found that the strings repeat for this length...
                            good_j = new_j;
                            // ...and with this length of the repeating pattern
                            good_k = k;
                            debug!("Best j and k is now {} and {}", good_j, good_k);
                        }
                        j += k;
                        continue;
                    } else {
                        // We know that it is no point in trying with more than `m`
                        if j == 0 {
                            k = m;
                            debug!("Setting new k to m: {}", k);
                        } else {
                            k -= 1;
                        }
                        debug!("Trying next k");
                        continue 'k_loop;
                    }
                }

                // if we exit the j-loop normally, try the next smaller `k`
                k -= 1;
            }

            // from `good_j` and `good_k` we know the repeat for a large
            // number of strings. The very last repeat length should not
            // be assigned, since it can be much longer if a new test is
            // done
            let mut m = 0;
            while m + good_k
                < usize::try_from(good_j).expect("we just converted the other way around")
                && i + m < nvals
            {
                // compute how many we actually repeat
                let repeat = (good_j as usize - m).min(nvals);
                // pack: low 8 bits = good_k, high bits = repeat
                nrepeat[i + m] = (good_k as u32) | ((repeat as u32) << 8);

                m += good_k;
            }

            // If no repetition was found for this value, signal that here
            if nrepeat[i] == 0 {
                // 257 == 1<<8 | 1
                nrepeat[i + m] = 257;
            }
        }
    }

    // Sort cyclic shift matrix
    bwt_sort(&mut indices, nvals, vals, &nrepeat);
    (0, 0)
}

/// c version: ptngc_bwt_merge_sort_inner
fn bwt_sort(indices: &mut [usize], nvals: usize, vals: &[u32], nrepeat: &[u32]) {
    // Only sort the [0..nvals] portion
    indices[..nvals].sort_by(|&ia, &ib| compare_index(ia, ib, nvals, vals, nrepeat));
}

fn compare_index(
    mut i1: usize,
    mut i2: usize,
    nvals: usize,
    vals: &[u32],
    nrepeat: &[u32],
) -> Ordering {
    let mut compared = 0;
    while compared < nvals {
        // If we have repeating patterns, we might be able to start the
        // comparison later in the string
        // Do we have a repeating pattern? If so are
        // the repeating patterns the same length?
        let packed1 = nrepeat[i1] as usize;
        let packed2 = nrepeat[i2] as usize;
        let repeat1 = packed1 >> 8;
        let k1 = packed1 & 0xFF;
        let repeat2 = packed2 >> 8;
        let k2 = packed2 & 0xFF;

        if repeat1 > 1 && repeat2 > 1 && k1 == k2 {
            // fast lexicographic compare of the next block of size k1
            let ord = vals
                .iter()
                .cycle()
                .skip(i1)
                .take(k1)
                .cmp(vals.iter().cycle().skip(i2).take(k1));
            if ord != Ordering::Equal {
                return ord;
            }
            // blocks were identical → skip ahead by min(repeat1, repeat2)
            let skip = repeat1.min(repeat2);
            i1 = (i1 + skip) % nvals;
            i2 = (i2 + skip) % nvals;
            compared += skip;
        } else {
            // single-element fallback
            let a = vals[i1];
            let b = vals[i2];
            if a < b {
                return Ordering::Less;
            }
            if a > b {
                return Ordering::Greater;
            }
            // advance each by one (cyclically)
            i1 = (i1 + 1) % nvals;
            i2 = (i2 + 1) % nvals;
            compared += 1;
        }
    }
    Ordering::Equal
}

/// Coding 32 bit ints in sequences of 16 bit ints. Worst case the output is `3*nvals` long
pub(crate) fn ptngc_comp_conv_to_vals16(vals: &[u32], vals16: &mut [u32]) -> usize {
    let mut j = 0;

    for val in vals {
        if *val <= 0x7FFF {
            vals16[j] = *val;
            j += 1;
        } else {
            let lo = (val & 0x7FFF) | 0x8000;
            let hi = val >> 15;
            vals16[j] = lo;
            j += 1;

            if hi <= 0x7FFF {
                vals16[j] = hi;
                j += 1;
            } else {
                let lohi = (hi & 0x7FFF) | 0x8000;
                let hihi = hi >> 15;
                vals16[j] = lohi;
                j += 1;
                vals16[j] = hihi;
                j += 1;
            }
        }
    }

    j
}

pub(crate) fn ptngc_comp_conv_from_vals16(vals16: &[u32], nvals16: usize, vals: &mut [u32]) -> i32 {
    let mut i: usize = 0;
    let mut j = 0;
    while i < usize::try_from(nvals16).expect("usize from i32") {
        if vals16[i] <= 0x7FFF {
            vals[j] = vals16[i];
            j += 1;
            i += 1;
        } else {
            let lo = vals16[i];
            i += 1;
            let hi = vals16[i];
            i += 1;

            if hi <= 0x7FFF {
                vals[j] = (lo & 0x7FFF) | (hi << 15);
                j += 1;
            } else {
                let hihi = vals16[i];
                i += 1;
                vals[j] = (lo & 0x7FFF) | ((hi & 0x7FFF) << 15) | (hihi << 30);
                j += 1;
            }
        }
    }
    i32::try_from(j).expect("i32 from usize")
}

#[cfg(test)]
mod sorting {
    use super::*;

    // in the following, some of the tests use bitshifts
    // to construct the `nrepeat` vecs. that's to easier test the
    // repeat logic. so when we write
    //            (repeat << 8) | k
    //
    //    [ repeat (…bits…) ][      k (8 bits) ]
    //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //          nrepeat entry

    fn run_case(vals: &[u32], nrepeat: &[u32]) -> Vec<usize> {
        let n = vals.len();
        let mut idx: Vec<usize> = (0..n).collect();
        bwt_sort(&mut idx, n, vals, nrepeat);
        idx
    }

    #[test]
    fn simple_in_order() {
        let vals = [1, 2, 3];
        let nrepeat = vec![0u32; vals.len()]; // no repeats
        let sorted = run_case(&vals, &nrepeat);
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn rotated_sequence() {
        // rotations: [3,1,2], [1,2,3], [2,3,1]
        let vals = [3, 1, 2];
        let nrepeat = vec![0u32; vals.len()]; // no repeats
        let sorted = run_case(&vals, &nrepeat);
        // lex order: [1,2,3](@1), [2,3,1](@2), [3,1,2](@0)
        assert_eq!(sorted, vec![1, 2, 0]);
    }

    #[test]
    fn all_equal() {
        let vals = [7, 7, 7, 7];
        let nrepeat = vec![0u32; vals.len()]; // no repeats
        let sorted = run_case(&vals, &nrepeat);
        // stable sort must preserve original [0,1,2,3]
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn full_repeated_pattern() {
        let vals = [1, 2, 1, 2, 1, 2];
        let nrepeat = [(3 << 8) | 2; 6];
        let sorted = run_case(&vals, &nrepeat);
        assert_eq!(sorted, vec![0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn partial_repeats_wrap() {
        let vals = [3, 4, 5, 3, 4];
        let nrepeat = [(2 << 8) | 3, 0, 0, (1 << 8) | 2, 0];
        let sorted = run_case(&vals, &nrepeat);
        assert_eq!(sorted, vec![3, 0, 4, 1, 2]);
    }

    #[test]
    fn mismatched_k() {
        let vals = [0, 1, 0, 1, 0, 2, 0, 2];
        let nrepeat = [(3 << 8) | 2, 0, 0, 0, 0, (2 << 8) | 3, 0, 0];
        let sorted = run_case(&vals, &nrepeat);
        assert_eq!(sorted, vec![0, 2, 6, 4, 1, 3, 7, 5]);
    }

    #[test]
    fn k_zero_malformed() {
        let vals = [5, 6, 7];
        let nrepeat = [(5 << 8), 0, 0];
        let sorted = run_case(&vals, &nrepeat);
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn overshoot_skip() {
        let vals = [9, 8, 7, 6];
        let nrepeat = [(10 << 8) | 1; 4];
        let sorted = run_case(&vals, &nrepeat);
        assert_eq!(sorted, vec![3, 2, 1, 0]);
    }
}

#[cfg(test)]
mod conv_to_vals16 {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init_logger() {
        INIT.call_once(|| {
            env_logger::builder().is_test(true).try_init().ok();
        });
    }

    #[test]
    fn single_small_values() {
        init_logger();

        let vals = [0x1234, 0x7FFF, 0x0000, 0x0001];
        let mut vals16 = [0u32; 20];

        let nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut vals16);

        assert_eq!(nvals16, 4);
        assert_eq!(vals16[0], 0x1234);
        assert_eq!(vals16[1], 0x7FFF);
        assert_eq!(vals16[2], 0x0000);
        assert_eq!(vals16[3], 0x0001);
    }

    #[test]
    fn two_chunk_values() {
        init_logger();

        // Test value 0x12345678
        // lo = (0x12345678 & 0x7FFF) | 0x8000 = 0x5678 | 0x8000 = 0xD678
        // hi = 0x12345678 >> 15 = 0x2468A (≤ 0x7FFF, so stored directly)
        let vals = [0x12345678];
        let mut vals16 = [0u32; 20];

        let nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut vals16);

        assert_eq!(nvals16, 2);
        let expected_lo = (0x12345678 & 0x7FFF) | 0x8000;
        let expected_hi = 0x12345678 >> 15;
        assert_eq!(vals16[0], expected_lo);
        assert_eq!(vals16[1], expected_hi);
    }

    #[test]
    fn three_chunk_values() {
        init_logger();

        // Test large value 0x80000000 (needs 3 chunks)
        // lo = (0x80000000 & 0x7FFF) | 0x8000 = 0x0000 | 0x8000 = 0x8000
        // hi = 0x80000000 >> 15 = 0x100000 (> 0x7FFF, needs splitting)
        // lohi = (0x100000 & 0x7FFF) | 0x8000 = 0x0000 | 0x8000 = 0x8000
        // hihi = 0x100000 >> 15 = 0x2000
        let vals = [0x80000000];
        let mut vals16 = [0u32; 20];

        let nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut vals16);

        assert_eq!(nvals16, 3);
        let expected_lo = 0x8000;
        let hi = 0x80000000 >> 15; // 0x100000
        let expected_lohi = (hi & 0x7FFF) | 0x8000; // 0x8000
        let expected_hihi = hi >> 15; // 0x2000

        assert_eq!(vals16[0], expected_lo);
        assert_eq!(vals16[1], expected_lohi);
        assert_eq!(vals16[2], expected_hihi);
    }

    #[test]
    fn boundary_values() {
        init_logger();

        let vals = [
            0x7FFF,     // Maximum single chunk
            0x8000,     // Minimum two-chunk
            0x3FFFFFFF, // Maximum two-chunk (hi = 0x7FFF)
            0x40000000, // Minimum three-chunk (hi = 0x8000)
        ];
        let mut vals16 = [0u32; 20];

        let _ = ptngc_comp_conv_to_vals16(&vals, &mut vals16);

        // vals[0] = 0x7FFF -> single chunk
        assert_eq!(vals16[0], 0x7FFF);

        // vals[1] = 0x8000 -> two chunks
        assert_eq!(vals16[1], 0x8000);
        assert_eq!(vals16[2], 0x8000 >> 15); // 0x0001

        // vals[2] = 0x3FFFFFFF -> two chunks (hi = 0x7FFF exactly)
        let idx = 3;
        assert_eq!(vals16[idx], (0x3FFFFFFF & 0x7FFF) | 0x8000); // 0xFFFF
        assert_eq!(vals16[idx + 1], 0x3FFFFFFF >> 15); // 0x7FFF

        // vals[3] = 0x40000000 -> three chunks (hi = 0x8000 exactly)
        let idx = idx + 2;
        let hi_val3 = 0x40000000 >> 15; // 0x8000
        assert_eq!(vals16[idx], 0x8000);
        assert_eq!(vals16[idx + 1], (hi_val3 & 0x7FFF) | 0x8000); // 0x8000
        assert_eq!(vals16[idx + 2], hi_val3 >> 15); // 0x0001
    }

    #[test]
    fn mixed_values() {
        init_logger();

        let vals = [
            0x1234,     // Single chunk
            0x12345678, // Two chunks
            0x5555,     // Single chunk
            0x80000001, // Three chunks
        ];
        let mut vals16 = [0u32; 20];

        let nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut vals16);

        // Should produce: 1 + 2 + 1 + 3 = 7 chunks
        assert_eq!(nvals16, 7);

        // First value (single)
        assert_eq!(vals16[0], 0x1234);

        // Second value (two chunks)
        assert_eq!(vals16[1], (0x12345678 & 0x7FFF) | 0x8000);
        assert_eq!(vals16[2], 0x12345678 >> 15);

        // Third value (single)
        assert_eq!(vals16[3], 0x5555);

        // Fourth value (three chunks)
        let hi_val4 = 0x80000001 >> 15;
        assert_eq!(vals16[4], (0x80000001 & 0x7FFF) | 0x8000);
        assert_eq!(vals16[5], (hi_val4 & 0x7FFF) | 0x8000);
        assert_eq!(vals16[6], hi_val4 >> 15);
    }

    #[test]
    fn empty_input() {
        init_logger();

        let vals: [u32; 0] = [];
        let mut vals16 = [0u32; 20];

        let nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut vals16);

        assert_eq!(nvals16, 0);
    }

    #[test]
    fn test_max_values() {
        init_logger();

        let vals = [0xFFFFFFFF];
        let mut vals16 = [0u32; 20];

        let nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut vals16);

        assert_eq!(nvals16, 3);

        // lo = (0xFFFFFFFF & 0x7FFF) | 0x8000 = 0x7FFF | 0x8000 = 0xFFFF
        // hi = 0xFFFFFFFF >> 15 = 0x1FFFF
        // lohi = (0x1FFFF & 0x7FFF) | 0x8000 = 0x7FFF | 0x8000 = 0xFFFF
        // hihi = 0x1FFFF >> 15 = 0x3
        assert_eq!(vals16[0], 0xFFFF);
        assert_eq!(vals16[1], 0xFFFF);
        assert_eq!(vals16[2], 0x3);
    }

    #[test]
    fn compression_efficiency() {
        init_logger();

        // Test various ranges to show compression behavior
        let test_cases = [
            (0x1000, 1),     // Small value -> 1 chunk
            (0x10000, 2),    // Medium value -> 2 chunks
            (0x1000000, 2),  // Large 2-chunk value -> 2 chunks
            (0x40000000, 3), // Large 3-chunk value -> 3 chunks
        ];

        for (val, expected_chunks) in test_cases {
            let vals = [val];
            let mut vals16 = [0u32; 10];

            let nvals16 = ptngc_comp_conv_to_vals16(&vals, &mut vals16);
            assert_eq!(
                nvals16, expected_chunks,
                "Value 0x{:X} should produce {} chunks",
                val, expected_chunks
            );
        }
    }
}

#[cfg(test)]
mod roundtrip {
    use crate::bwlzh::{ptngc_comp_conv_from_vals16, ptngc_comp_conv_to_vals16};

    #[test]
    fn roundtrip_single_values() {
        let original = [0x1234, 0x7FFF, 0x0000, 0x0001, 0x5A5A];
        let mut vals16 = [0u32; 20];
        let mut reconstructed = [0u32; 10];

        // Compress
        let nvals16 = ptngc_comp_conv_to_vals16(&original, &mut vals16);

        // Decompress
        let nvals_reconstructed =
            ptngc_comp_conv_from_vals16(&vals16[..nvals16 as usize], nvals16, &mut reconstructed);

        // Verify round-trip
        assert_eq!(nvals_reconstructed, 5);
        for i in 0..5 {
            assert_eq!(
                reconstructed[i], original[i],
                "Mismatch at index {}: 0x{:X} != 0x{:X}",
                i, original[i], reconstructed[i]
            );
        }
    }

    #[test]
    fn roundtrip_two_chunk_values() {
        let original = [
            0x8000,     // Minimum two-chunk
            0x12345678, // Typical two-chunk
            0x3FFFFFFF, // Maximum two-chunk
        ];
        let mut vals16 = [0u32; 20];
        let mut reconstructed = [0u32; 10];

        // Compress
        let nvals16 = ptngc_comp_conv_to_vals16(&original, &mut vals16);

        // Decompress
        let nvals_reconstructed =
            ptngc_comp_conv_from_vals16(&vals16[..nvals16 as usize], nvals16, &mut reconstructed);

        // Verify round-trip
        assert_eq!(nvals_reconstructed, 3);
        for i in 0..3 {
            assert_eq!(reconstructed[i], original[i]);
            println!("  0x{:08X} -> 0x{:08X} ✓", original[i], reconstructed[i]);
        }
    }

    #[test]
    fn test_roundtrip_three_chunk_values() {
        let original = [
            0x40000000, // Minimum three-chunk
            0x80000000, // Large three-chunk
            0xFFFFFFFF, // Maximum value
        ];
        let mut vals16 = [0u32; 20];
        let mut reconstructed = [0u32; 10];

        // Compress
        let nvals16 = ptngc_comp_conv_to_vals16(&original, &mut vals16);

        // Decompress
        let nvals_reconstructed =
            ptngc_comp_conv_from_vals16(&vals16[..nvals16 as usize], nvals16, &mut reconstructed);

        // Verify round-trip
        assert_eq!(nvals_reconstructed, 3);
        for i in 0..3 {
            assert_eq!(reconstructed[i], original[i]);
            println!("  0x{:08X} -> 0x{:08X} ✓", original[i], reconstructed[i]);
        }
    }

    #[test]
    fn roundtrip_mixed_values() {
        let original = [
            0x0001,     // Single chunk
            0x7FFF,     // Single chunk (boundary)
            0x8000,     // Two chunks (boundary)
            0x12345678, // Two chunks
            0x3FFFFFFF, // Two chunks (boundary)
            0x40000000, // Three chunks (boundary)
            0xABCDEF12, // Three chunks
            0xFFFFFFFF, // Three chunks (max)
        ];
        let mut vals16 = [0u32; 30];
        let mut reconstructed = [0u32; 20];

        // Compress
        let nvals16 = ptngc_comp_conv_to_vals16(&original, &mut vals16);

        println!("  Compression: {} values -> {} chunks", 8, nvals16);

        // Decompress
        let nvals_reconstructed =
            ptngc_comp_conv_from_vals16(&vals16[..nvals16 as usize], nvals16, &mut reconstructed);

        // Verify round-trip
        assert_eq!(nvals_reconstructed, 8);
        for i in 0..8 {
            assert_eq!(reconstructed[i], original[i]);
            println!("  0x{:08X} -> 0x{:08X} ✓", original[i], reconstructed[i]);
        }
    }

    #[test]
    fn test_roundtrip_stress() {
        // Generate test data covering all ranges
        let mut original = Vec::new();

        // Add single-chunk values (0 to 0x7FFF)
        for i in 0..100 {
            original.push(i * 327); // Spread across range
        }

        // Add two-chunk values (0x8000 to 0x3FFFFFFF)
        for i in 0..100 {
            original.push(0x8000 + i * 655360); // Spread across range
        }

        // Add three-chunk values (0x40000000 to 0xFFFFFFFF)
        for i in 0..100 {
            original.push(0x40000000 + i * 1234567);
        }

        let mut vals16 = vec![0u32; 3000]; // Worst case: all 3-chunk = 3x expansion
        let mut reconstructed = vec![0u32; 1000];

        // Compress
        let nvals16 = ptngc_comp_conv_to_vals16(&original, &mut vals16);

        // Decompress
        let nvals_reconstructed =
            ptngc_comp_conv_from_vals16(&vals16[..nvals16 as usize], nvals16, &mut reconstructed);

        // Verify round-trip
        assert_eq!(nvals_reconstructed as usize, original.len());
        for i in 0..original.len() {
            assert_eq!(
                reconstructed[i], original[i],
                "MISMATCH at index {}: 0x{:08X} != 0x{:08X}",
                i, original[i], reconstructed[i]
            );
        }
    }

    #[test]
    fn roundtrip_edge_cases() {
        let original = [
            0x00000000, // Minimum value
            0x00007FFF, // Maximum single-chunk
            0x00008000, // Minimum two-chunk
            0x3FFFFFFF, // Maximum two-chunk
            0x40000000, // Minimum three-chunk
            0xFFFFFFFF, // Maximum value
            // Powers of 2 boundaries
            0x00000001, 0x00000002, 0x00000004, 0x00008000, 0x00010000, 0x00020000, 0x40000000,
            0x80000000, // Bit pattern tests
            0x55555555, // Alternating bits
            0xAAAAAAAA, // Alternating bits (inverted)
            0x12345678, // Mixed pattern
            0x87654321, // Reversed pattern
        ];

        let mut vals16 = [0u32; 50];
        let mut reconstructed = [0u32; 30];

        // Compress
        let nvals16 = ptngc_comp_conv_to_vals16(&original, &mut vals16);

        // Decompress
        let nvals_reconstructed =
            ptngc_comp_conv_from_vals16(&vals16[..nvals16 as usize], nvals16, &mut reconstructed);

        // Verify round-trip
        assert_eq!(nvals_reconstructed as usize, original.len());
        for i in 0..original.len() {
            assert_eq!(reconstructed[i], original[i]);
        }
    }

    #[test]
    fn test_roundtrip_compression_ratio() {
        let cases = [
            vec![1, 2, 3, 4, 5, 100, 1000, 0x7FFF],
            vec![0x10000, 0x100000, 0x1000000, 0x3FFFFFFF],
            vec![0x40000000, 0x80000000, 0xC0000000, 0xFFFFFFFF],
        ];

        for case in &cases {
            let mut vals16 = vec![0u32; 20];
            let mut reconstructed = vec![0u32; 10];

            // Compress
            let nvals16 = ptngc_comp_conv_to_vals16(case, &mut vals16);

            // Decompress
            let nvals_reconstructed = ptngc_comp_conv_from_vals16(
                &vals16[..nvals16 as usize],
                nvals16,
                &mut reconstructed,
            );

            // Verify round-trip
            assert_eq!(nvals_reconstructed as usize, case.len());
            for (idx, recon) in reconstructed.iter().enumerate() {
                assert_eq!(*recon, case[idx]);
            }
        }
    }

    #[test]
    fn roundtrip_empty() {
        let original: [u32; 0] = [];
        let mut vals16 = [0u32; 10];
        let mut reconstructed = [0u32; 10];

        // Compress
        let nvals16 = ptngc_comp_conv_to_vals16(&original, &mut vals16);

        // Decompress
        let nvals_reconstructed =
            ptngc_comp_conv_from_vals16(&vals16[..nvals16 as usize], nvals16, &mut reconstructed);

        // Verify round-trip
        assert_eq!(nvals16, 0);
        assert_eq!(nvals_reconstructed, 0);
    }

    #[test]
    fn compression_properties() {
        // Test that compression is deterministic
        let test_value = 0x12345678u32;
        let mut vals16_a = [0u32; 10];
        let mut vals16_b = [0u32; 10];

        let nvals16_a = ptngc_comp_conv_to_vals16(&[test_value], &mut vals16_a);
        let nvals16_b = ptngc_comp_conv_to_vals16(&[test_value], &mut vals16_b);

        assert_eq!(nvals16_a, nvals16_b);
        for i in 0..nvals16_a as usize {
            assert_eq!(vals16_a[i], vals16_b[i]);
        }

        // Test that decompression is also deterministic
        let mut reconstructed_a = [0u32; 10];
        let mut reconstructed_b = [0u32; 10];

        let nvals_a = ptngc_comp_conv_from_vals16(
            &vals16_a[..nvals16_a as usize],
            nvals16_a,
            &mut reconstructed_a,
        );
        let nvals_b = ptngc_comp_conv_from_vals16(
            &vals16_b[..nvals16_b as usize],
            nvals16_b,
            &mut reconstructed_b,
        );

        assert_eq!(nvals_a, nvals_b);
        assert_eq!(reconstructed_a[0], reconstructed_b[0]);
        assert_eq!(reconstructed_a[0], test_value);
    }
}
