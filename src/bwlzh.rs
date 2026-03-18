use std::cmp::Ordering;

use log::debug;

use crate::{
    huffmem::{
        ptngc_comp_get_huff_algo_name, ptngc_comp_huff_buflen, ptngc_comp_huff_compress_verbose,
    },
    lz77::ptngc_comp_to_lz77,
    mtf::ptngc_comp_conv_to_mtf_partial3,
    rle::ptngc_comp_conv_to_rle,
    utils::copy_bytes,
};
const MAX_VALS_PER_BLOCK: usize = 200000;

// TODO: enable these as compile-time features?
const PARTIAL_MTF3: bool = true;
const PARTIAL_MTF: bool = false;
pub const N_HUFFMAN_ALGO: usize = 3;

pub(crate) const fn bwlzh_get_buflen(nvals: usize) -> usize {
    132000 + nvals * 8 + 12 * ((nvals + MAX_VALS_PER_BLOCK) / MAX_VALS_PER_BLOCK)
}

/// Compress the integers (positive, small integers are preferable) using bwlzh compression. The
/// unsigned char *output should be allocated to be able to hold worst case. You can obtain this
/// length conveniently by calling `comp_get_buflen()`
pub(crate) fn bwlzh_compress(vals: &[u32], nvals: usize, output: &mut [u8]) -> usize {
    bwlzh_compress_gen(vals, nvals, output, true)
}

pub(crate) fn bwlzh_compress_no_lz77(vals: &[u32], nvals: usize, output: &mut [u8]) -> usize {
    bwlzh_compress_gen(vals, nvals, output, false)
}

pub(crate) fn bwlzh_compress_gen(
    vals: &[u32],
    nvals: usize,
    output: &mut [u8],
    enable_lz77: bool,
) -> usize {
    let mut outdata = 0;
    let mut valsleft;
    let mut valstart;
    let mut thisvals;
    let mut huffalgo;
    let mut bwlzhhuff = vec![0; ptngc_comp_huff_buflen(3 * nvals)];
    let total_len = MAX_VALS_PER_BLOCK * 18;
    let mut tmpmem = vec![0u32; total_len];

    let (vals16, rest) = tmpmem.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (bwt, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (mtf, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (rle, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let mut nrle = 0;
    let (offsets, lens) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);

    // TODO: enable feature flag "partial-mtf3"
    let mut mtf3 = vec![0; MAX_VALS_PER_BLOCK * 3 * 3];

    debug!("Number of input values: {nvals}");

    // Store the number of real values in the whole block
    let bytes = (nvals as u32).to_le_bytes(); // [u8; 4]
    output[outdata..outdata + 4].copy_from_slice(&bytes);
    outdata += 4;

    valsleft = nvals;
    valstart = 0;

    while valsleft > 0 {
        let mut reducealgo = 1; // Reduce algo is LZ77
        if !enable_lz77 {
            reducealgo = 0;
        }
        thisvals = valsleft;
        if thisvals > MAX_VALS_PER_BLOCK {
            thisvals = MAX_VALS_PER_BLOCK;
        }
        valsleft -= thisvals;
        debug!("Creating vals16 block from {thisvals} values");
        let nvals16 = ptngc_comp_conv_to_vals16(&vals[valstart..], vals16);
        valstart += thisvals;

        debug!("Resulting vals16 values: {nvals16}");
        debug!("BWT");
        let bwt_index = ptngc_comp_to_bwt(vals16, nvals16, bwt);

        // Store the number of real values in this block
        // Store the number of nvals16 in this block
        // Store the BWT index
        for &word in &[thisvals, nvals16, bwt_index] {
            let bytes = (word as u32).to_le_bytes(); // [low, …, high]
            output[outdata..outdata + 4].copy_from_slice(&bytes);
            outdata += 4;
        }

        debug!("MTF");
        // TODO: make sure the other PARTIAL_MTFs stuff is implemented
        if PARTIAL_MTF3 {
            ptngc_comp_conv_to_mtf_partial3(bwt, nvals16, &mut mtf3);
            for imtfinner in 0..3 {
                debug!("Doing partial MTF: {imtfinner}");

                for j in 0..nvals16 {
                    mtf[j] = u32::from(mtf3[imtfinner * nvals16 + j]);
                }
                // } else if PARTIAL_MTF {
                //     ptngc_comp_conv_to_mtf_partial(bwt, mtf);
                // } else {
                //     ptngc_comp_canonical_dict(&mut dict);
                //     ptngc_comp_conv_to_mtf(bwt, &dict, mtf);
                // }

                let mut nlens = 0usize;
                let mut noffsets = 0usize;
                if reducealgo == 1 {
                    debug!("LZ77");

                    let lz77_result = ptngc_comp_to_lz77(&mtf[..nvals16]);
                    for (out, &src) in rle
                        .iter_mut()
                        .take(lz77_result.data.len())
                        .zip(lz77_result.data.iter())
                    {
                        *out = src as u32;
                    }
                    nrle = lz77_result.data.len();
                    nlens = lz77_result.lengths.len();
                    for (dst, &src) in lens[..nlens].iter_mut().zip(lz77_result.lengths.iter()) {
                        *dst = src as u32;
                    }
                    noffsets = lz77_result.offsets.len();
                    for (dst, &src) in offsets[..noffsets]
                        .iter_mut()
                        .zip(lz77_result.offsets.iter())
                    {
                        *dst = src as u32;
                    }

                    debug!("Resulting LZ77 values: {nrle}");
                    debug!("Resulting LZ77 lens: {nlens}");
                    debug!("Resulting LZ77 offsets: {noffsets}");

                    // block that is "if 0"
                    if nlens < 2 {
                        reducealgo = 0;
                    }
                }
                if reducealgo == 0 {
                    debug!("RLE");

                    // Do RLE. For any repetitive characters
                    let result = ptngc_comp_conv_to_rle(&mtf[..nvals16], 1);
                    rle[..result.len()].copy_from_slice(&result);
                    nrle = result.len();
                    debug!("Resulting RLE values: {}", rle.len());
                }

                // reducealgo: RLE == 0, LZ77 == 1
                output[outdata] = reducealgo;
                outdata += 1;

                debug!("Huffman");
                huffalgo = -1;
                let mut bwlzhhufflen = 0;
                let mut huffdatalen = 0;
                let mut nhufflen = vec![0; N_HUFFMAN_ALGO];
                ptngc_comp_huff_compress_verbose(
                    &mut rle[..nrle],
                    &mut bwlzhhuff,
                    &mut bwlzhhufflen,
                    &mut huffdatalen,
                    &mut nhufflen,
                    &mut huffalgo,
                    true,
                );

                // if verbose
                debug!("Huffman data length is {huffdatalen}");
                for (i, nhl) in nhufflen.iter().enumerate().take(N_HUFFMAN_ALGO) {
                    debug!(
                        "Huffman dictionary for algorithm {} is {}",
                        ptngc_comp_get_huff_algo_name(i).expect("algo name"),
                        nhl - huffdatalen
                    )
                }

                // Store the number of huffman values in this block
                copy_bytes(nrle as u32, output, &mut outdata);

                // Store the size of the huffman block
                copy_bytes(bwlzhhufflen as u32, output, &mut outdata);

                // Store the huffman block
                let bwlzhufflen_usize = usize::try_from(bwlzhhufflen).expect("i32 to usize");
                output[outdata..outdata + bwlzhufflen_usize]
                    .copy_from_slice(&bwlzhhuff[..bwlzhufflen_usize]);
                outdata += bwlzhufflen_usize;

                if reducealgo == 1 {
                    // Store the number of values in this block
                    copy_bytes(noffsets as u32, output, &mut outdata);

                    if noffsets > 0 {
                        debug!("Huffman for offsets");

                        huffalgo = -1;
                        ptngc_comp_huff_compress_verbose(
                            &mut offsets[..noffsets],
                            &mut bwlzhhuff,
                            &mut bwlzhhufflen,
                            &mut huffdatalen,
                            &mut nhufflen,
                            &mut huffalgo,
                            true,
                        );

                        // if verbose
                        debug!("Huffman data length is {huffdatalen} B");
                        for (i, nhl) in nhufflen.iter().enumerate().take(N_HUFFMAN_ALGO) {
                            debug!(
                                "Huffman dictionary for algorithm {} is {}",
                                ptngc_comp_get_huff_algo_name(i).expect("algo name"),
                                nhl - huffdatalen
                            )
                        }
                        debug!(
                            "Resulting algorithm: {}. Size={} B",
                            ptngc_comp_get_huff_algo_name(
                                usize::try_from(huffalgo).expect("usize to i32")
                            )
                            .expect("algo name"),
                            bwlzhhufflen
                        );

                        // If huffman was bad for these offsets, just store the offsets as pairs
                        if bwlzhhufflen < i32::try_from(noffsets).expect("i32 from usize") * 2 {
                            output[outdata] = 0;

                            // Store the size of the huffman block
                            copy_bytes(bwlzhhufflen as u32, output, &mut outdata);

                            // Store the huffman block
                            output[outdata
                                ..outdata + usize::try_from(bwlzhhufflen).expect("usize from i32")]
                                .copy_from_slice(
                                    &bwlzhhuff
                                        [..usize::try_from(bwlzhhufflen).expect("usize from i32")],
                                );
                            outdata += usize::try_from(bwlzhhufflen).expect("usize from i32");
                        } else {
                            output[outdata] = 1;
                            outdata += 1;
                            for os in &mut offsets[..noffsets] {
                                output[outdata] = (*os & 0xFF) as u8;
                                outdata += 1;
                                output[outdata] = ((*os >> 8) & 0xFF) as u8;
                                outdata += 1;
                            }

                            // if verbose
                            debug!("Store raw offsets: {} B", noffsets * 2);
                        }
                    }

                    // if verbose
                    debug!("Huffman for lengths");

                    huffalgo = -1;
                    ptngc_comp_huff_compress_verbose(
                        &mut lens[..nlens],
                        &mut bwlzhhuff,
                        &mut bwlzhhufflen,
                        &mut huffdatalen,
                        &mut nhufflen,
                        &mut huffalgo,
                        true,
                    );

                    // if verbose
                    debug!("Huffman data length is {huffdatalen} B");
                    for (i, nhl) in nhufflen.iter().enumerate().take(N_HUFFMAN_ALGO) {
                        debug!(
                            "Huffman dictionary for algorithm {} is {}",
                            ptngc_comp_get_huff_algo_name(i).expect("algo name"),
                            nhl - huffdatalen
                        );
                    }
                    debug!(
                        "Resulting algorithm: {}. Size={} B",
                        ptngc_comp_get_huff_algo_name(
                            usize::try_from(huffalgo).expect("usize from i32")
                        )
                        .expect("algo name"),
                        bwlzhhufflen
                    );

                    // Store the number of values in this block
                    copy_bytes(nlens as u32, output, &mut outdata);

                    // Store the size of the huffman block
                    copy_bytes(bwlzhhufflen as u32, output, &mut outdata);

                    // Store the huffman block
                    let bwlzhhufflen_usize = usize::try_from(bwlzhhufflen).expect("usize from i32");
                    output[outdata..outdata + bwlzhhufflen_usize]
                        .copy_from_slice(&bwlzhhuff[..bwlzhhufflen_usize]);
                    outdata += usize::try_from(bwlzhhufflen).expect("usize from i32");
                }
            }
        }
    }

    outdata
}

fn bwlzh_decompress_gen(input: &mut [u8], nvals: i32, vals: &mut [u8]) {
    let mut max_vals_per_block = MAX_VALS_PER_BLOCK as i32;
    let mut inpdata = 0;

    let total_len = MAX_VALS_PER_BLOCK * 18;
    let mut tmpmem = vec![0u32; total_len];
    let bwlzhhuff = vec![0; ptngc_comp_huff_buflen(3 * nvals as usize)];
    let (mut vals16, mut rest) = tmpmem.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (mut bwt, mut rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (mut mtf, mut rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (mut rle, mut rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    let (mut offsets, mut lens) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
    // TODO: enable feature flag "partial-mtf3"

    debug!("Number of input values: {nvals}");

    // Read the number of real values in the whole block
    let nvalsfile = i32::from_le_bytes(input[inpdata..inpdata + 4].try_into().unwrap());
    inpdata += 4;

    if nvalsfile != nvals {
        panic!(
            "BWLZH: The number of values found in the file is different from the number of values expected."
        );
    }

    let mut valsleft = nvals;
    let mut valstart = 0;

    while valsleft != 0 {
        let valsnew;
        let reducealgo;
        // Read the number of real values in this block
        let thisvals = i32::from_le_bytes(input[inpdata..inpdata + 4].try_into().unwrap());
        inpdata += 4;

        valsleft -= thisvals;

        // Read the number of nvals16 values in this block
        let nvals16 = i32::from_le_bytes(input[inpdata..inpdata + 4].try_into().unwrap());
        inpdata += 4;

        // Read the BWT index
        let bwt_index = i32::from_le_bytes(input[inpdata..inpdata + 4].try_into().unwrap());
        inpdata += 4;

        if thisvals > max_vals_per_block {
            // More memory must be allocated for decompression
            max_vals_per_block = thisvals;
            debug!(
                "Allocating more memory: {} B",
                (max_vals_per_block as usize * 15 * size_of::<u32>()) as i32
            );
            tmpmem = vec![0u32; total_len];
            (vals16, rest) = tmpmem.split_at_mut(MAX_VALS_PER_BLOCK * 3);
            (bwt, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
            (mtf, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
            (rle, rest) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
            (offsets, lens) = rest.split_at_mut(MAX_VALS_PER_BLOCK * 3);
            // TODO: enable feature flag "partial-mtf3"
        }

        // TODO: enable feature flag "partial-mtf3"
        reducealgo = input[inpdata] as i32;
        inpdata += 1;

        // Read the number of huffman values in this block
        let nrle = i32::from_le_bytes(input[inpdata..inpdata + 4].try_into().unwrap());
        inpdata += 4;

        // Read the size of the huffman block
        let bwlzhhufflen = i32::from_le_bytes(input[inpdata..inpdata + 4].try_into().unwrap());
        inpdata += 4;

        debug!("Decompressing huffman block of length {bwlzhhufflen}");

        // Decompress the huffman block
        ptngc_comp_huff_decompress(&input[inpdata..], bwlzhhufflen, rle);
        inpdata += bwlzhhufflen;
    }
}

pub(crate) fn bwlzh_decompress(input: &mut [u8], nvals: i32, vals: &mut [u8]) {
    bwlzh_decompress_gen(input, nvals, vals)
}

/// Burrows-Wheeler transform
pub(crate) fn ptngc_comp_to_bwt(vals: &[u32], nvals: usize, output: &mut [u32]) -> usize {
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
            let mut best_repeat: Option<(usize, usize)> = None;
            let kmax = 16;
            // Track repeating patterns.
            // k=1 corresponds to AAAAA...
            // k=2 corresponds to ABABAB...
            // k=3 corresponds to ABCABCABCABC...
            // k=4 corresponds to ABCDABCDABCD...
            // etc.
            let mut k = kmax;
            'k_loop: while k >= 1 {
                debug!("Trying k={k} at i={i}");

                // for j = k; j < maxrepeat; j += k
                let mut j = k;
                while j < maxrepeat {
                    debug!("Trying j={j} at i={i} for k={k}");

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
                        let new_j = if j + k > maxrepeat { j } else { j + k };
                        match best_repeat {
                            None => {
                                best_repeat = Some((new_j, k));
                                debug!("Best j and k is now {new_j} and {k}");
                            }
                            Some((best_j, best_k)) => {
                                if new_j > best_j || (new_j == best_j && k < best_k) {
                                    best_repeat = Some((new_j, k));
                                    debug!("Best j and k is now {new_j} and {k}");
                                }
                            }
                        }
                        j += k;
                        continue;
                    } else {
                        // We know that it is no point in trying with more than `m`
                        if j == 0 {
                            k = m;
                            debug!("Setting new k to m: {k}");
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
            if let Some((good_j, good_k)) = best_repeat {
                let mut m = 0;
                while m + good_k < good_j && i + m < nvals {
                    let repeat = (good_j - m).min(nvals);
                    nrepeat[i + m] = (good_k as u32) | ((repeat as u32) << 8);
                    m += good_k;
                }
                if nrepeat[i] == 0 {
                    nrepeat[i + m] = 257;
                }
            } else {
                nrepeat[i] = 257;
            }
        }
    }

    // Sort cyclic shift matrix
    bwt_sort(&mut indices, nvals, vals, &nrepeat);

    // which one is the original string?
    let index = indices.iter().position(|x| *x == 0).unwrap_or(nvals);

    // Form output
    for (i, &idx) in indices.iter().enumerate() {
        let lastchar = if idx == 0 { nvals - 1 } else { idx - 1 };
        output[i] = vals[lastchar];
    }

    index
}

/// Burrows-Wheeler inverse transform
///
/// c version: Ptngc_comp_from_bwt
pub(crate) fn inverse_bwt(input: &[u32], index: usize, vals: &mut [u32]) {
    // Straightforward from the Burrows-Wheeler paper (page 13).
    let nvals = input.len();
    let mut c = vec![0u32; 0x10000];
    let mut p = vec![0u32; nvals];

    for (i, &val) in input.iter().enumerate() {
        p[i] = c[val as usize];
        c[val as usize] += 1;
    }

    let mut sum = 0u32;
    for count in c.iter_mut() {
        sum += *count;
        *count = sum - *count;
    }

    let mut idx = index;
    for i in (0..nvals).rev() {
        let val = input[idx];
        vals[i] = val;
        idx = (p[idx] + c[val as usize]) as usize;
    }
}

/// c version: ptngc_bwt_merge_sort_inner
pub(crate) fn bwt_sort(indices: &mut [usize], nvals: usize, vals: &[u32], nrepeat: &[u32]) {
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
    let mut i = 0;
    while i < nvals {
        // If we have repeating patterns, we might be able to start the
        // comparison later in the string
        // Do we have a repeating pattern? If so are
        // the repeating patterns the same length?
        let repeat1 = nrepeat[i1] >> 8;
        let k1 = nrepeat[i1] & 0xFF;
        let repeat2 = nrepeat[i2] >> 8;
        let k2 = nrepeat[i2] & 0xFF;

        if repeat1 > 1 && repeat2 > 1 && k1 == k2 {
            // Yes. Compare the repeating patterns
            for j in 0..k1 {
                let v1 = vals[(i1 + j as usize) % nvals];
                let v2 = vals[(i2 + j as usize) % nvals];
                if v1 < v2 {
                    return Ordering::Less;
                }
                if v1 > v2 {
                    return Ordering::Greater;
                }
            }

            // The repeating patters are equal. Skip as far as we can before continuing
            let skip = std::cmp::min(repeat1, repeat2);
            i1 = (i1 + skip as usize) % nvals;
            i2 = (i2 + skip as usize) % nvals;
            i += skip as usize;
        } else {
            // single-element fallback
            if vals[i1] < vals[i2] {
                return Ordering::Less;
            }
            if vals[i1] > vals[i2] {
                return Ordering::Greater;
            }
            // advance each by one (cyclically)
            i1 = (i1 + 1) % nvals;
            i2 = (i2 + 1) % nvals;
            i += 1;
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
    while i < nvals16 {
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
            ptngc_comp_conv_from_vals16(&vals16[..nvals16], nvals16, &mut reconstructed);

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
            ptngc_comp_conv_from_vals16(&vals16[..nvals16], nvals16, &mut reconstructed);

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
            ptngc_comp_conv_from_vals16(&vals16[..nvals16], nvals16, &mut reconstructed);

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
            ptngc_comp_conv_from_vals16(&vals16[..nvals16], nvals16, &mut reconstructed);

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
            ptngc_comp_conv_from_vals16(&vals16[..nvals16], nvals16, &mut reconstructed);

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
            ptngc_comp_conv_from_vals16(&vals16[..nvals16], nvals16, &mut reconstructed);

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
            for idx in 0..case.len() {
                assert_eq!(reconstructed[idx], case[idx]);
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
            ptngc_comp_conv_from_vals16(&vals16[..nvals16], nvals16, &mut reconstructed);

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
        for i in 0..nvals16_a {
            assert_eq!(vals16_a[i], vals16_b[i]);
        }

        // Test that decompression is also deterministic
        let mut reconstructed_a = [0u32; 10];
        let mut reconstructed_b = [0u32; 10];

        let nvals_a =
            ptngc_comp_conv_from_vals16(&vals16_a[..nvals16_a], nvals16_a, &mut reconstructed_a);
        let nvals_b =
            ptngc_comp_conv_from_vals16(&vals16_b[..nvals16_b], nvals16_b, &mut reconstructed_b);

        assert_eq!(nvals_a, nvals_b);
        assert_eq!(reconstructed_a[0], reconstructed_b[0]);
        assert_eq!(reconstructed_a[0], test_value);
    }
}

#[cfg(test)]
mod test_bwlzh {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init_logger() {
        INIT.call_once(|| {
            env_logger::builder().is_test(true).try_init().ok();
        });
    }

    #[test]
    fn it_works() {
        init_logger();

        let vals = vec![1, 2, 3, 4, 5];
        let mut output = vec![0; 4 + bwlzh_get_buflen(vals.len())];
        let nvals = vals.len();

        let noutput = bwlzh_compress_gen(&vals, nvals, &mut output, true);
        let expected_output = vec![
            5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 29, 0, 0, 0, 1, 0, 5, 0,
            0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 198, 192, 7, 0, 0, 4, 0, 0, 7, 0, 0, 8, 162, 138, 32, 0,
            3, 0, 0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 152, 6, 0, 0, 3, 0,
            0, 2, 0, 0, 134, 40, 128, 0, 3, 0, 0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1,
            0, 0, 0, 152, 6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128,
        ];
        assert_eq!(noutput, 126);
        assert_eq!(expected_output, output[..noutput]);
    }

    #[test]
    fn without_lz77() {
        init_logger();

        let vals = vec![1, 2, 3, 4, 5];
        let mut output = vec![0; 4 + bwlzh_get_buflen(vals.len())];
        let nvals = vals.len();

        let noutput = bwlzh_compress_gen(&vals, nvals, &mut output, false);
        let expected_output = vec![
            5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 29, 0, 0, 0, 1, 0, 5, 0,
            0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 198, 192, 7, 0, 0, 4, 0, 0, 7, 0, 0, 8, 162, 138, 32, 0,
            3, 0, 0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 152, 6, 0, 0, 3, 0,
            0, 2, 0, 0, 134, 40, 128, 0, 3, 0, 0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1,
            0, 0, 0, 152, 6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128,
        ];
        assert_eq!(noutput, 126);
        assert_eq!(expected_output, output[..noutput]);
    }

    #[test]
    fn repeated_values() {
        init_logger();

        let vals = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
        let mut output = vec![0; 4 + bwlzh_get_buflen(vals.len())];
        let nvals = vals.len();

        let noutput = bwlzh_compress_gen(&vals, nvals, &mut output, true);
        let expected_output = vec![
            9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 30, 0, 0, 0, 1, 0, 9, 0,
            0, 0, 9, 0, 0, 0, 3, 0, 0, 0, 156, 66, 112, 7, 0, 0, 5, 0, 0, 5, 0, 0, 141, 20, 113,
            68, 0, 4, 0, 0, 0, 27, 0, 0, 0, 1, 0, 4, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 140, 6, 0, 0,
            3, 0, 0, 2, 0, 0, 134, 40, 128, 0, 4, 0, 0, 0, 27, 0, 0, 0, 1, 0, 4, 0, 0, 0, 4, 0, 0,
            0, 1, 0, 0, 0, 140, 6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128,
        ];
        assert_eq!(noutput, 127);
        assert_eq!(expected_output, output[..noutput]);
    }

    #[test]
    fn single_value() {
        init_logger();

        let vals = vec![42];
        let mut output = vec![0; 4 + bwlzh_get_buflen(vals.len())];
        let nvals = vals.len();

        let noutput = bwlzh_compress_gen(&vals, nvals, &mut output, true);
        let expected_output = vec![
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 31, 0, 0, 0, 1, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 10, 0, 0, 1, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 8, 64, 0,
            1, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0,
            2, 0, 0, 33, 0, 1, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
            4, 0, 0, 1, 0, 0, 2, 0, 0, 33,
        ];
        assert_eq!(noutput, 124);
        assert_eq!(expected_output, output[..noutput]);
    }

    #[test]
    fn empty_input() {
        init_logger();

        let vals = vec![];
        let mut output = vec![0; 4 + bwlzh_get_buflen(vals.len())];
        let nvals = vals.len();

        let noutput = bwlzh_compress_gen(&vals, nvals, &mut output, true);
        let expected_output = vec![0, 0, 0, 0];
        assert_eq!(noutput, 4);
        assert_eq!(expected_output, output[..noutput]);
    }

    #[test]
    fn large_values() {
        init_logger();

        let vals = vec![0xFFFFFFFF, 0x80000000, 0x12345678, 0xABCDEF00];
        let mut output = vec![0; 4 + bwlzh_get_buflen(vals.len())];
        let nvals = vals.len();

        let noutput = bwlzh_compress_gen(&vals, nvals, &mut output, true);
        let expected_output = vec![
            4, 0, 0, 0, 4, 0, 0, 0, 11, 0, 0, 0, 10, 0, 0, 0, 0, 11, 0, 0, 0, 57, 0, 0, 0, 1, 2,
            11, 0, 0, 0, 11, 0, 0, 0, 5, 0, 0, 0, 61, 250, 197, 89, 24, 5, 1, 0, 10, 0, 0, 40, 0,
            0, 12, 0, 0, 8, 0, 0, 6, 0, 0, 117, 215, 86, 172, 66, 95, 73, 124, 23, 228, 18, 248, 6,
            0, 0, 134, 56, 228, 71, 32, 0, 11, 0, 0, 0, 58, 0, 0, 0, 1, 2, 11, 0, 0, 0, 11, 0, 0,
            0, 5, 0, 0, 0, 185, 250, 80, 232, 200, 5, 1, 0, 10, 0, 0, 44, 0, 0, 13, 0, 0, 8, 0, 0,
            6, 0, 0, 57, 177, 57, 64, 176, 162, 88, 81, 35, 213, 47, 22, 240, 6, 0, 0, 138, 40,
            164, 71, 32, 0, 4, 0, 0, 0, 27, 0, 0, 0, 1, 0, 4, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 44,
            6, 0, 0, 3, 0, 0, 2, 0, 0, 138, 24, 128,
        ];
        assert_eq!(noutput, 185);
        assert_eq!(expected_output, output[..noutput]);
    }

    #[test]
    fn alternating_pattern() {
        init_logger();

        let vals = vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
        let mut output = vec![0; 4 + bwlzh_get_buflen(vals.len())];
        let nvals = vals.len();

        let noutput = bwlzh_compress_gen(&vals, nvals, &mut output, true);
        let expected_output = vec![
            10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 1, 0, 8,
            0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 203, 32, 6, 0, 0, 3, 0, 0, 4, 0, 0, 133, 18, 32, 0, 4,
            0, 0, 0, 27, 0, 0, 0, 1, 0, 4, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 76, 6, 0, 0, 3, 0, 0,
            2, 0, 0, 134, 40, 128, 0, 4, 0, 0, 0, 27, 0, 0, 0, 1, 0, 4, 0, 0, 0, 4, 0, 0, 0, 1, 0,
            0, 0, 76, 6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128,
        ];
        assert_eq!(noutput, 125);
        assert_eq!(expected_output, output[..noutput]);
    }

    #[test]
    fn sequential() {
        init_logger();

        let vals = (0..20).collect::<Vec<_>>();
        let mut output = vec![0; 4 + bwlzh_get_buflen(vals.len())];
        let nvals = vals.len();

        let noutput = bwlzh_compress_gen(&vals, nvals, &mut output, true);
        let expected_output = vec![
            20, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 49, 0, 0, 0, 1, 0,
            20, 0, 0, 0, 20, 0, 0, 0, 11, 0, 0, 0, 192, 18, 52, 86, 120, 154, 189, 111, 157, 247,
            240, 18, 0, 0, 19, 0, 0, 21, 0, 0, 18, 73, 36, 146, 73, 36, 146, 73, 36, 146, 203, 44,
            178, 203, 32, 0, 5, 0, 0, 0, 27, 0, 0, 0, 1, 0, 5, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 38,
            6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128, 0, 5, 0, 0, 0, 27, 0, 0, 0, 1, 0, 5, 0, 0, 0,
            5, 0, 0, 0, 1, 0, 0, 0, 38, 6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128,
        ];
        assert_eq!(noutput, 146);
        assert_eq!(expected_output, output[..noutput]);
    }
}
