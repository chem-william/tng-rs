use crate::{
    bwlzh::{bwlzh_compress, bwlzh_compress_no_lz77, bwlzh_decompress, bwlzh_get_buflen},
    widemuldiv::{ptngc_largeint_add, ptngc_largeint_div, ptngc_largeint_mul},
    xtc2::{ptngc_find_magic_index, ptngc_magic},
};

#[allow(clippy::excessive_precision)]
pub(crate) const IFLIPGAINCHECK: f64 = 0.890_898_718_140_339_27; /*  1./(2**(1./6)) */

// Maximum number of large atoms for large RLE
const MAX_LARGE_RLE: usize = 1024;
// Maximum number of small atoms in one group
const MAX_SMALL_RLE: usize = 12;

// Sequence instructions
const INSTR_DEFAULT: u32 = 0;
const INSTR_SMALL_RUNLENGTH: u32 = 1;
const INSTR_ONLY_LARGE: u32 = 2;
const INSTR_ONLY_SMALL: u32 = 3;
const INSTR_FLIP: u32 = 4;
const INSTR_LARGE_RLE: u32 = 5;
const INSTR_LARGE_DIRECT: u32 = 6;
const INSTR_LARGE_INTRA_DELTA: u32 = 7;
const INSTR_LARGE_INTER_DELTA: u32 = 8;

const MAXINSTR: usize = 9;

// How much larger can the direct frame deltas for the small triplets be
// and be accepted anyway as better than the intra/inter frame deltas.
// For better instructions/RLEs
const THRESHOLD_INTRA_INTER_DIRECT: f64 = 1.5;

// How much larger can the intra frame deltas for the small triplets be
// and be accepted anyway as better than the inter frame deltas
const THRESHOLD_INTER_INTRA: f64 = 5.0;

// Difference in indices used for determining whether to store as
// large or small. A fun detail in this compression algorithm is that
// if everything works fine, large can often be smaller than small, or
// at least not as large as is large in magic.c. This is a key idea of
// xtc3.
const QUITE_LARGE: u32 = 3;
const IS_LARGE: u32 = 6;

// The base_compress routine first compresses all x coordinates, then
// y and finally z. The bases used for each can be different. The
// MAXBASEVALS value determines how many coordinates are compressed
// into a single number. Only resulting whole bytes are dealt with for
// simplicity. MAXMAXBASEVALS is the insanely large value to accept
// files written with that value. BASEINTERVAL determines how often a
// new base is actually computed and stored in the output
// file. MAXBASEVALS*BASEINTERVAL values are stored using the same
// base in BASEINTERVAL different integers. Note that the primarily
// the decompression using a large MAXBASEVALS becomes very slow.
const MAXMAXBASEVALS: usize = 16384;
const MAXBASEVALS: usize = 24;
const BASEINTERVAL: usize = 8;

#[derive(Debug)]
struct Xtc3Context {
    instructions: Vec<u32>,
    ninstr: i32,
    rle: Vec<u32>,
    large_direct: Vec<u32>,
    large_intra_delta: Vec<u32>,
    large_inter_delta: Vec<u32>,
    smallintra: Vec<u32>,
    minint: [i32; 3],
    maxint: [i32; 3],
    has_large: usize,
    // Large cache
    has_large_ints: [u32; MAX_LARGE_RLE * 3],
    // What kind of type this large is
    has_large_type: [i32; MAX_LARGE_RLE],
    current_large_type: i32,
}

impl Default for Xtc3Context {
    fn default() -> Self {
        Self {
            instructions: Vec::new(),
            ninstr: 0,
            rle: Vec::new(),
            large_direct: Vec::new(),
            large_intra_delta: Vec::new(),
            large_inter_delta: Vec::new(),
            smallintra: Vec::new(),
            minint: [0; 3],
            maxint: [0; 3],
            has_large_ints: [0; MAX_LARGE_RLE * 3],
            has_large_type: [0; MAX_LARGE_RLE],
            has_large: 0,
            current_large_type: 0,
        }
    }
}

// Modifies three integer values for better compression of water
pub(crate) fn swap_ints(input: &[i32; 3], output: &mut [i32; 3]) {
    output[0] = input[0] + input[1];
    output[1] = -input[1];
    output[2] = input[1] + input[2];
}

pub(crate) fn swap_is_better(input: &[i32], minint: &[i32; 3]) -> (u32, u32) {
    let mut normal_max = 0;
    let mut swapped_max = 0;
    let mut normal = [0; 3];
    let mut swapped = [0; 3];

    for i in 0..3 {
        normal[0] = input[i].wrapping_sub(minint[i]);
        normal[1] = input[3 + i].wrapping_sub(input[i]); // minint[i]-minint[i] cancels out
        normal[2] = input[6 + i].wrapping_sub(input[3 + i]); // minint[i]-minint[i] cancels out
        swap_ints(&normal, &mut swapped);

        for j in 1..3 {
            if positive_int(normal[j]) > normal_max {
                normal_max = positive_int(normal[j]);
            }

            if positive_int(swapped[j]) > swapped_max {
                swapped_max = positive_int(swapped[j]);
            }
        }
    }

    if normal_max == 0 {
        normal_max = 1;
    }

    if swapped_max == 0 {
        swapped_max = 1;
    }

    (normal_max, swapped_max)
}

impl Xtc3Context {
    fn swapdecide(&mut self, input: &[i32], swapatoms: &mut bool) {
        let mut didswap = false;

        let (normal, swapped) = swap_is_better(input, &self.minint);
        // We have to determine if it is worth to change the behaviour.
        // If diff is positive it means that it is worth something to
        // swap. But it costs 4 bits to do the change. If we assume that
        // we gain 0.17 bit by the swap per value, and the runlength>2
        // for four molecules in a row, we gain something. So check if we
        // gain at least 0.17 bits to even attempt the swap.

        if (swapped < normal) && ((f64::from(swapped) / f64::from(normal)).abs() < IFLIPGAINCHECK)
            || ((normal < swapped)
                && (f64::from(normal) / f64::from(swapped)).abs() < IFLIPGAINCHECK)
        {
            if swapped < normal {
                if !*swapatoms {
                    *swapatoms = true;
                    didswap = true;
                }
            } else if *swapatoms {
                *swapatoms = false;
                didswap = true;
            }
        }

        if didswap {
            self.instructions.push(INSTR_FLIP);
        }
    }

    fn write_three_large(&mut self, i: usize) {
        let target = match self.current_large_type {
            0 => &mut self.large_direct,
            1 => &mut self.large_intra_delta,
            _ => &mut self.large_inter_delta,
        };
        target.extend_from_slice(&self.has_large_ints[i * 3..i * 3 + 3]);
    }

    fn large_instruction_change(&mut self, i: usize) {
        if self.has_large_type[i] != self.current_large_type {
            self.current_large_type = self.has_large_type[i];
            let instr = match self.current_large_type {
                0 => INSTR_LARGE_DIRECT,
                1 => INSTR_LARGE_INTRA_DELTA,
                _ => INSTR_LARGE_INTER_DELTA,
            };
            // insert_value_in_array
            self.instructions.push(instr);
        }
    }

    fn flush_large(&mut self, n: usize) {
        let mut i = 0;

        while i < n {
            // If the first large is of a different kind than the currently used we must
            // emit an "instruction" to change the large type
            self.large_instruction_change(i);

            // How many large of the same kind in a row
            let mut j = 0;
            while i + j < n && self.has_large_type[i + j] == self.has_large_type[i] {
                j += 1;
            }

            if j < 3 {
                for k in 0..j {
                    self.instructions.push(INSTR_ONLY_LARGE);
                    self.write_three_large(i + k);
                }
            } else {
                self.instructions.push(INSTR_LARGE_RLE);
                self.rle.push(u32::try_from(j).expect("u32 from usize"));
                for k in 0..j {
                    self.write_three_large(i + k);
                }
            }
            i += j;
        }
        if self.has_large - n != 0 {
            for i in 0..(self.has_large - n) {
                self.has_large_type[i] = self.has_large_type[i + n];
                for j in 0..3 {
                    self.has_large_ints[i * 3 + j] = self.has_large_ints[(i + n) * 3 + j];
                }
            }
        }
        // Number of remaining large atoms in buffer
        self.has_large -= n;
    }

    fn buffer_large(
        &mut self,
        input: &mut [i32],
        inpdata: usize,
        natoms: usize,
        intradelta_ok: bool,
    ) {
        let mut intradelta = [0; 3];
        let mut interdelta = [0; 3];
        let frame = inpdata / (natoms * 3);
        let atomframe = inpdata % (natoms * 3);
        // If it is full we must write them all
        if self.has_large == MAX_LARGE_RLE {
            // Flush all
            self.flush_large(self.has_large);
        }
        // Find out which is the best choice for the large integer. Direct coding, or some
        //kind of delta coding?
        // First create direct coding.
        let direct = [
            (input[inpdata].wrapping_sub(self.minint[0])) as u32,
            (input[inpdata + 1].wrapping_sub(self.minint[1])) as u32,
            (input[inpdata + 2].wrapping_sub(self.minint[2])) as u32,
        ];
        let mut minlen = compute_intlen(direct.as_slice());
        let mut best_type = 0; // Direct

        // in c code: #if 1
        // Then try intra coding if we can
        if intradelta_ok && atomframe >= 3 {
            intradelta[0] = positive_int(input[inpdata].wrapping_sub(input[inpdata - 3]));
            intradelta[1] = positive_int(input[inpdata + 1].wrapping_sub(input[inpdata - 2]));
            intradelta[2] = positive_int(input[inpdata + 2].wrapping_sub(input[inpdata - 1]));
            let thislen = compute_intlen(intradelta.as_slice());
            if thislen * THRESHOLD_INTRA_INTER_DIRECT < minlen {
                minlen = thislen;
                best_type = 1; // Intra delta
            }
        }
        // in c code: #endif
        // in c code: #if 1
        // Then try inter coding if we can
        if frame > 0 {
            interdelta[0] = positive_int(input[inpdata].wrapping_sub(input[inpdata - natoms * 3]));
            interdelta[1] =
                positive_int(input[inpdata + 1].wrapping_sub(input[inpdata - natoms * 3 + 1]));
            interdelta[2] =
                positive_int(input[inpdata + 2].wrapping_sub(input[inpdata - natoms * 3 + 2]));
            let thislen = compute_intlen(interdelta.as_slice());
            if thislen * THRESHOLD_INTRA_INTER_DIRECT < minlen {
                best_type = 2; // Inter delta
            }
        }
        // in c code: #endif
        self.has_large_type[self.has_large] = best_type;
        let src = match best_type {
            0 => &direct,
            1 => &intradelta,
            2 => &interdelta,
            _ => unreachable!(),
        };
        self.has_large_ints[self.has_large * 3..self.has_large * 3 + 3].copy_from_slice(src);
        self.has_large += 1;
    }
}

// How many bytes are needed to store `n` values in `base` base
fn base_bytes(base: u32, n: usize) -> usize {
    let mut largeint = [0; MAXBASEVALS + 1];
    let mut largeint_tmp = [0; MAXBASEVALS + 1];
    let mut numbytes = 0;

    for i in 0..n {
        if i != 0 {
            ptngc_largeint_mul(base, &largeint, &mut largeint_tmp, n + 1);
            largeint[..n + 1].copy_from_slice(&largeint_tmp[..n + 1]);
        }
        ptngc_largeint_add(base - 1, &mut largeint, n + 1);
    }

    for (i, &item) in largeint.iter().enumerate() {
        if item > 0 {
            for j in 0..4 {
                if (item >> (j * 8)) & 0xFF > 0 {
                    numbytes = i * 4 + j + 1;
                }
            }
        }
    }
    numbytes
}

fn copy_single_byte(largeint: &mut [u32], j: usize, output: &mut Vec<u8>) {
    let ilarge = j / 4;
    let ibyte = j % 4;
    let byte = ((largeint[ilarge] >> (ibyte * 8)) & 0xFF) as u8;
    output.push(byte);
}

fn base_compress(data: &[u32], len: usize) -> (Vec<u8>, usize) {
    let mut largeint = [0; MAXBASEVALS + 1];
    let mut largeint_tmp = [0; MAXBASEVALS + 1];
    let mut numbytes = 0;
    let mut output = Vec::with_capacity(len + 3);

    // Store the MAXBASEVALS value in the output
    output.extend_from_slice(&(MAXBASEVALS as u16).to_le_bytes());
    // Store the BASEINTERVAL value in the output
    output.push(BASEINTERVAL as u8);

    for ixyz in 0..3 {
        let mut base = 0;
        let mut nvals = 0;
        let mut basegiven = 0;
        largeint.fill(0);

        for i in (ixyz..len).step_by(3) {
            if nvals == 0 {
                let mut basecheckvals = 0;
                if basegiven == 0 {
                    base = 0;
                    // Find the largest value for this particular coordinate
                    for k in (i..len).step_by(3) {
                        if data[k] > base {
                            base = data[k];
                        }
                        basecheckvals += 1;
                        if basecheckvals == MAXBASEVALS * BASEINTERVAL {
                            break;
                        }
                    }
                    // The base is one larger than the largest value
                    base += 1;
                    if base < 2 {
                        base = 2;
                    }
                    // Store the base in the output
                    output.extend_from_slice(&base.to_le_bytes());
                    basegiven = BASEINTERVAL;

                    // How many bytes is needed to store MAXBASEVALS values using this base?
                    numbytes = base_bytes(base, MAXBASEVALS);
                }
                basegiven -= 1;
            }
            if nvals != 0 {
                ptngc_largeint_mul(base, &largeint, &mut largeint_tmp, MAXBASEVALS + 1);
                largeint[..MAXBASEVALS + 1].copy_from_slice(&largeint_tmp);
            }
            ptngc_largeint_add(data[i], &mut largeint, MAXBASEVALS + 1);
            nvals += 1;
            if nvals == MAXBASEVALS {
                for j in 0..numbytes {
                    copy_single_byte(&mut largeint, j, &mut output);
                }
                nvals = 0;
                largeint.fill(0);
            }
        }
        if nvals > 0 {
            numbytes = base_bytes(base, nvals);
            for j in 0..numbytes {
                copy_single_byte(&mut largeint, j, &mut output);
            }
        }
    }

    let output_len = output.len();
    (output, output_len)
}

#[inline]
pub(crate) fn positive_int(item: i32) -> u32 {
    // Branchless zigzag: 0->0, 1->1, -1->2, 2->3, -2->4, ...
    // For positive i: 2*i - 1; for negative i: -2*i; for zero: 0
    if item > 0 {
        (item as u32) * 2 - 1
    } else {
        (item.wrapping_neg() as u32) * 2
    }
}

#[inline]
pub(crate) fn unpositive_int(val: i32) -> i32 {
    let mut s = (val + 1) / 2;
    if val % 2 == 0 {
        s = -s;
    }
    s
}

fn compute_intlen(ints: &[u32]) -> f64 {
    // /* The largest value. */
    ints.iter().copied().max().unwrap_or_default().into()
}

fn insert_batch(
    input_ptr: &[i32],
    ntriplets_left: usize,
    prevcoord: &[i32],
    encode_ints: &mut [i32],
    startenc: usize,
) -> usize {
    let mut tmp_prevcoord = [prevcoord[0], prevcoord[1], prevcoord[2]];

    for chunk in encode_ints.chunks_exact(3).take(startenc) {
        tmp_prevcoord[0] = tmp_prevcoord[0].wrapping_add(chunk[0]);
        tmp_prevcoord[1] = tmp_prevcoord[1].wrapping_add(chunk[1]);
        tmp_prevcoord[2] = tmp_prevcoord[2].wrapping_add(chunk[2]);
    }

    let total_triplets = (1 + MAX_SMALL_RLE).min(ntriplets_left);
    let start_idx = startenc * 3;
    let end_idx = total_triplets * 3;

    for (encode_chunk, input_chunk) in encode_ints[start_idx..end_idx]
        .chunks_exact_mut(3)
        .zip(input_ptr[start_idx..].chunks_exact(3))
    {
        encode_chunk[0] = input_chunk[0].wrapping_sub(tmp_prevcoord[0]);
        encode_chunk[1] = input_chunk[1].wrapping_sub(tmp_prevcoord[1]);
        encode_chunk[2] = input_chunk[2].wrapping_sub(tmp_prevcoord[2]);

        tmp_prevcoord = [input_chunk[0], input_chunk[1], input_chunk[2]];
    }
    end_idx
}

// It is "large" if we have to increase the small index quite a
// bit. Not so much to be rejected by the not very large check
// later.
pub(crate) fn is_quite_large(input: &[i32], small_index: u32, max_large_index: u32) -> bool {
    let predicate =
        ptngc_magic(usize::try_from(small_index + QUITE_LARGE).expect("usize from u32"));
    if small_index + QUITE_LARGE >= max_large_index {
        true
    } else {
        input
            .iter()
            .take(3)
            .any(|inp| positive_int(*inp) > predicate)
    }
}

fn output_int(output: &mut [u8], outdata: &mut usize, n: u32) {
    let bytes = n.to_le_bytes();
    output[*outdata..*outdata + 4].copy_from_slice(&bytes);
    *outdata += 4;
}

// Speed selects how careful to try to find the most efficient compression. The BWLZH algo is expensive!
// Speed <=2 always avoids BWLZH everywhere it is possible.
// Speed 3 and 4 and 5 use heuristics (check proportion of large value). This should mostly be safe.
// Speed 5 enables the LZ77 component of BWLZH.
// Speed 6 always tests if BWLZH is better and if it is uses it. This can be very slow.
pub(crate) fn ptngc_pack_array_xtc3(
    input: &mut [i32],
    length: &mut usize,
    natoms: usize,
    speed: &mut usize,
) -> (Vec<u8>, usize) {
    let mut outdata = 0;
    let ntriplets = *length / 3;
    let mut runlength = 0; // Initial runlength. "Stupidly" set to zero for simplicity and explicity

    // Initial guess is that we should not swap the first two atoms in each large+small transition
    let mut swapatoms = false;
    // Wether swapping was actually done
    let mut didswap;
    let mut inpdata = 0;
    let mut ntriplets_left = ntriplets;
    let mut large_index = [0; 3];
    let mut encode_ints: [i32; 3 + MAX_SMALL_RLE * 3] = [0; 3 + MAX_SMALL_RLE * 3];
    let mut refused = 0;
    // let mut base_buf;

    let mut xtc3_context = Xtc3Context::default();
    xtc3_context.maxint.copy_from_slice(&input[..3]);
    xtc3_context.minint.copy_from_slice(&input[..3]);

    // Values of speed should be sane
    *speed = (*speed).clamp(1, 6);

    // Allocate enough memory for output
    let mut output = if *length < 48 {
        vec![0; 48 * 8]
    } else {
        vec![0; 8 * (*length)]
    };

    for i in 0..ntriplets {
        for j in 0..3 {
            if input[i * 3 + j] > xtc3_context.maxint[j] {
                xtc3_context.maxint[j] = input[i * 3 + j];
            }

            if input[i * 3 + j] < xtc3_context.minint[j] {
                xtc3_context.minint[j] = input[i * 3 + j];
            }
        }
    }

    large_index[0] = ptngc_find_magic_index(
        (xtc3_context.maxint[0]
            .wrapping_sub(xtc3_context.minint[0])
            .wrapping_add(1)) as u32,
    );
    large_index[1] = ptngc_find_magic_index(
        (xtc3_context.maxint[1]
            .wrapping_sub(xtc3_context.minint[1])
            .wrapping_add(1)) as u32,
    );
    large_index[2] = ptngc_find_magic_index(
        (xtc3_context.maxint[2]
            .wrapping_sub(xtc3_context.minint[2])
            .wrapping_add(1)) as u32,
    );
    let max_large_index = *large_index.iter().max().expect("large_index to be init");

    // Guess initial small index
    let mut small_index = max_large_index / 2;

    // Find the largest value that is not large. Not large is half index of large
    let max_small = ptngc_magic(usize::try_from(small_index).expect("usize from u32"));
    let mut intmax = 0;
    for item in &input[..*length] {
        let s = positive_int(*item);
        if s > intmax && s < max_small {
            intmax = s;
        }
    }
    // This value is not critical, since if I guess wrong, the code will
    // just insert instructions to increase this value
    small_index = ptngc_find_magic_index(intmax);

    output_int(
        &mut output,
        &mut outdata,
        positive_int(xtc3_context.minint[0]),
    );
    output_int(
        &mut output,
        &mut outdata,
        positive_int(xtc3_context.minint[1]),
    );
    output_int(
        &mut output,
        &mut outdata,
        positive_int(xtc3_context.minint[2]),
    );

    // Initial prevcoord is the minimum integers
    let mut prevcoord = [
        xtc3_context.minint[0],
        xtc3_context.minint[1],
        xtc3_context.minint[2],
    ];

    while ntriplets_left > 0 {
        // If only less than three atoms left we just write them all as large integers. Here no swapping is done!
        if ntriplets_left < 3 {
            // TODO: does this loop not only run once?
            xtc3_context.buffer_large(input, inpdata, natoms, true);
            inpdata += 3;
            ntriplets_left -= 1;
            xtc3_context.flush_large(xtc3_context.has_large); // Flush all
        } else {
            let mut min_runlength = 0;
            let mut largest_runlength_base;
            let mut largest_runlength_index;
            let mut new_runlength;
            let mut new_small_index;
            let mut iter_runlength;
            let mut iter_small_index;
            let mut rle_index_dep;

            didswap = false;
            // Insert the next batch of integers to be encoded into the buffer
            let mut nencode = insert_batch(
                &input[inpdata..],
                ntriplets_left,
                prevcoord.as_slice(),
                encode_ints.as_mut_slice(),
                0,
            );

            // First we must decide if the next value is large (does not reasonably fit in current small encoding)
            // Also, if we have not written any values yet, we must begin by writing a large atom. */
            if (inpdata == 0)
                || (is_quite_large(&encode_ints, small_index, max_large_index))
                || refused > 0
            {
                // If any of the next two atoms are large we should probably write them as large and not swap them
                let mut no_swap = false;
                if is_quite_large(&encode_ints[3..], small_index, max_large_index)
                    || is_quite_large(&encode_ints[6..], small_index, max_large_index)
                {
                    no_swap = true;
                }

                if !no_swap {
                    // If doing inter-frame coding results in smaller values we should not do any swapping either
                    let frame = inpdata / (natoms * 3);
                    if frame > 0 {
                        let mut delta = [0; 3];
                        let mut delta2 = [0; 3];

                        delta[0] =
                            positive_int(input[inpdata + 3] - input[inpdata - natoms * 3 + 3]);
                        delta[1] =
                            positive_int(input[inpdata + 4] - input[inpdata - natoms * 3 + 4]);
                        delta[2] =
                            positive_int(input[inpdata + 5] - input[inpdata - natoms * 3 + 5]);
                        delta2[0] = positive_int(encode_ints[3]);
                        delta2[1] = positive_int(encode_ints[4]);
                        delta2[2] = positive_int(encode_ints[5]);

                        if compute_intlen(&delta) * THRESHOLD_INTER_INTRA < compute_intlen(&delta2)
                        {
                            delta[0] =
                                positive_int(input[inpdata + 6] - input[inpdata - natoms * 3 + 6]);
                            delta[1] =
                                positive_int(input[inpdata + 7] - input[inpdata - natoms * 3 + 7]);
                            delta[2] =
                                positive_int(input[inpdata + 8] - input[inpdata - natoms * 3 + 8]);

                            delta2[0] = positive_int(encode_ints[6]);
                            delta2[1] = positive_int(encode_ints[7]);
                            delta2[2] = positive_int(encode_ints[8]);

                            if compute_intlen(&delta) * THRESHOLD_INTER_INTRA
                                < compute_intlen(&delta2)
                            {
                                no_swap = true;
                            }
                        }
                    }
                }

                if !no_swap {
                    // Next we must decide if we should swap the first two values
                    xtc3_context.swapdecide(&input[inpdata..], &mut swapatoms);

                    // If we should do the integer swapping manipulation we should do it now
                    if swapatoms {
                        didswap = true;
                        for i in 0..3 {
                            let swapped_input = [
                                input[inpdata + i],
                                input[inpdata + 3 + i].wrapping_sub(input[inpdata + i]),
                                input[inpdata + 6 + i].wrapping_sub(input[inpdata + 3 + i]),
                            ];
                            let mut output = [0; 3];
                            swap_ints(&swapped_input, &mut output);
                            encode_ints[i] = output[0];
                            encode_ints[3 + i] = output[1];
                            encode_ints[6 + i] = output[2];
                        }
                        // We have swapped atoms, so the minimum run-length is 2
                        min_runlength = 2;
                    }
                }
                // Cache large value for later possible combination with a sequence of small integers
                if swapatoms && didswap {
                    // This is a swapped integer, so `inpdata` is one atom later and intra coding is not ok
                    xtc3_context.buffer_large(input, inpdata + 3, natoms, false);

                    for ienc in 0..3 {
                        prevcoord[ienc] = input[inpdata + 3 + ienc];
                    }
                } else {
                    xtc3_context.buffer_large(input, inpdata, natoms, true);
                    prevcoord.copy_from_slice(&input[inpdata..(3 + inpdata)]);
                }

                // We have written a large integer so we have one less atoms to worry about
                inpdata += 3;
                ntriplets_left -= 1;

                refused = 0;

                // Insert the next batch of integers to be encoded into the buffer
                if swapatoms && didswap {
                    // Keep swapped values
                    for i in 0..2 {
                        for ienc in 0..3 {
                            encode_ints[i * 3 + ienc] = encode_ints[(i + 1) * 3 + ienc];
                        }
                    }
                }
                nencode = insert_batch(
                    &input[inpdata..],
                    ntriplets_left,
                    prevcoord.as_slice(),
                    encode_ints.as_mut_slice(),
                    min_runlength,
                );
            }

            // Here we should only have differences for the atom coordinates.
            // Convert the ints to positive ints
            for item in &mut encode_ints[..nencode] {
                // Match the C encoder, which stores the `positive_int` result in `int`
                // and therefore wraps when the value exceeds `i32::MAX`.
                *item = positive_int(*item) as i32;
            }
            // Now we must decide what base and runlength to do. If we have swapped atoms it will be
            // at least 2. If even the next atom is large, we will not do anything

            // Determine required base
            let largest_required_base = *encode_ints
                .iter()
                .take(min_runlength * 3)
                .filter(|&&x| x.is_positive())
                .max()
                .unwrap_or(&0);

            largest_runlength_base = *encode_ints
                .iter()
                .take((runlength * 3).min(nencode))
                .filter(|&&x| x.is_positive())
                .max()
                .unwrap_or(&0);

            let largest_required_index = ptngc_find_magic_index(largest_required_base as u32);
            largest_runlength_index = ptngc_find_magic_index(largest_runlength_base as u32);

            if largest_required_index < largest_runlength_index {
                new_runlength = min_runlength;
                new_small_index = largest_required_index;
            } else {
                new_runlength = runlength;
                new_small_index = largest_runlength_index;
            }

            // Only allow increase of runlength wrt min_runlength
            if new_runlength < min_runlength {
                new_runlength = min_runlength;
            }

            // If the current runlength is longer than the number of triplets left stop it from being so
            if new_runlength > ntriplets_left {
                new_runlength = ntriplets_left;
            }

            // We must at least try to get some small integers going
            if new_runlength == 0 {
                new_runlength = 1;
                new_small_index = small_index;
            }

            iter_runlength = new_runlength;
            iter_small_index = new_small_index;

            // Iterate to find optimal encoding and runlength
            loop {
                new_runlength = iter_runlength;
                new_small_index = iter_small_index;

                // What is the largest runlength we can do with the currently
                // selected encoding? Also the max supported runlength is MAX_SMALL_RLE triplets!
                let mut ienc = 0;
                while ienc < nencode && ienc < MAX_SMALL_RLE * 3 {
                    let test_index = ptngc_find_magic_index(encode_ints[ienc] as u32);
                    if test_index > new_small_index {
                        break;
                    }

                    ienc += 1;
                }
                if ienc / 3 > new_runlength {
                    iter_runlength = ienc / 3;
                }

                // How large encoding do we have to use?
                largest_runlength_base = 0;
                for &item in encode_ints.iter().take(iter_runlength * 3) {
                    if item > largest_runlength_base {
                        largest_runlength_base = item;
                    }
                }
                largest_runlength_index = ptngc_find_magic_index(largest_runlength_base as u32);
                if largest_runlength_index != new_small_index {
                    iter_small_index = largest_runlength_index;
                }

                // to emulate the do .. while construct in c, we put this if statement at the end
                if new_runlength == iter_runlength && new_small_index == iter_small_index {
                    break;
                }
            }

            // Verify that we got something good. We may have caught a
            // substantially larger atom. If so we should just bail
            // out and let the loop get on another lap. We may have a
            // minimum runlength though and then we have to fulfill
            // the request to write out these atoms!
            rle_index_dep = 0;
            if new_runlength < 3 {
                rle_index_dep = IS_LARGE;
            } else if new_runlength < 6 {
                rle_index_dep = QUITE_LARGE;
            }

            if min_runlength > 0
                || (new_small_index < small_index + IS_LARGE)
                    && (new_small_index + rle_index_dep < max_large_index)
                || (new_small_index + IS_LARGE < max_large_index)
            {
                // If doing inter-frame coding of large integers results
                // in smaller values than the small value we should not
                // produce a sequence of small values here.
                let frame = inpdata / (natoms * 3);
                let mut numsmaller = 0;

                if !swapatoms && frame > 0 {
                    for i in 0..new_runlength {
                        let delta = [
                            positive_int(
                                input[inpdata + i * 3] - input[inpdata - natoms * 3 + i * 3],
                            ),
                            positive_int(
                                input[inpdata + i * 3 + 1]
                                    - input[inpdata - natoms * 3 + i * 3 + 1],
                            ),
                            positive_int(
                                input[inpdata + i * 3 + 2]
                                    - input[inpdata - natoms * 3 + i * 3 + 2],
                            ),
                        ];
                        let delta2 = [
                            positive_int(encode_ints[i * 3]),
                            positive_int(encode_ints[i * 3 + 1]),
                            positive_int(encode_ints[i * 3 + 2]),
                        ];
                        if compute_intlen(&delta) * THRESHOLD_INTER_INTRA < compute_intlen(&delta2)
                        {
                            numsmaller += 1;
                        }
                    }
                }
                // Most of the values should become smaller, otherwise
                // we should encode them with intra coding.
                if !swapatoms && (numsmaller >= 2 * new_runlength / 3) {
                    // Put all the values in large arrays, instead of the small array
                    if new_runlength > 0 {
                        for i in 0..new_runlength {
                            xtc3_context.buffer_large(input, inpdata + i * 3, natoms, true);
                        }
                        for i in 0..3 {
                            prevcoord[i] = input[inpdata + (new_runlength - 1) * 3 + i];
                        }
                        inpdata += 3 * new_runlength;
                        ntriplets_left -= new_runlength;
                    }
                } else {
                    if new_runlength != runlength || new_small_index != small_index {
                        let mut change: i32 = new_small_index as i32 - small_index as i32;
                        // c code: had `new_small_index` as an `int` and did a "<=" check
                        if new_small_index == 0 {
                            change = 0;
                        }

                        if change < 0 {
                            for ixx in 0..new_runlength {
                                let mut rejected;
                                loop {
                                    let mut isum = 0.; // ints can be almost 32 bit so multiplication will overflow. So do doubles
                                    for ixyz in 0..3 {
                                        // `encode_ints` is already positive (and multiplied by 2 versus the original, just as magic ints)
                                        let id = f64::from(encode_ints[ixx * 3 + ixyz]);
                                        isum += id * id;
                                    }
                                    rejected = false;

                                    if isum
                                        > f64::from(ptngc_magic(
                                            usize::try_from((small_index as i32 + change) as u32)
                                                .expect("usize from u32"),
                                        )) * f64::from(ptngc_magic(
                                            usize::try_from((small_index as i32 + change) as u32)
                                                .expect("usize from u32"),
                                        ))
                                    {
                                        rejected = true;
                                        change += 1;
                                    }
                                    if !(change < 0 && rejected) {
                                        break;
                                    }
                                }
                                if change == 0 {
                                    break;
                                }
                            }
                        }

                        // Always accept the new small indices here
                        small_index = new_small_index;
                        // If we have a new runlength emit it
                        if runlength != new_runlength {
                            runlength = new_runlength;
                            xtc3_context.instructions.push(INSTR_SMALL_RUNLENGTH);
                            xtc3_context
                                .rle
                                .push(runlength.try_into().expect("u32 to usize"));
                        }
                    }
                    // If we have a large previous integer we can combine it with a sequence
                    if xtc3_context.has_large > 0 {
                        // If swapatoms is set to 1 but we did actually not
                        // do any swapping, we must first write out the
                        // large atom and then the small. If swapatoms is 1
                        // and we did swapping we can use the efficient
                        // encoding.
                        if swapatoms && !didswap {
                            // Flush all large atoms
                            xtc3_context.flush_large(xtc3_context.has_large);
                            xtc3_context.instructions.push(INSTR_ONLY_SMALL);
                        } else {
                            // Flush all large atoms but one!
                            if xtc3_context.has_large > 1 {
                                xtc3_context.flush_large(xtc3_context.has_large - 1);
                            }

                            // Here we must check if we should emit a large
                            // type change instruction
                            xtc3_context.large_instruction_change(0);
                            xtc3_context.instructions.push(INSTR_DEFAULT);
                            xtc3_context.write_three_large(0);
                            xtc3_context.has_large = 0;
                        }
                    } else {
                        xtc3_context.instructions.push(INSTR_ONLY_SMALL);
                    }
                    // Insert the small integers into the small integer array
                    for item in &encode_ints[..runlength * 3] {
                        xtc3_context.smallintra.push(*item as u32);
                    }
                    // Update `prevcoord`
                    for ienc in 0..runlength {
                        prevcoord[0] =
                            prevcoord[0].wrapping_add(unpositive_int(encode_ints[ienc * 3]));
                        prevcoord[1] =
                            prevcoord[1].wrapping_add(unpositive_int(encode_ints[ienc * 3 + 1]));
                        prevcoord[2] =
                            prevcoord[2].wrapping_add(unpositive_int(encode_ints[ienc * 3 + 2]));
                    }
                    inpdata += 3 * runlength;
                    ntriplets_left -= runlength;
                }
            } else {
                refused += 1;
            }
        }
    }

    // If we have large previous integers we must flush them now
    if xtc3_context.has_large > 0 {
        xtc3_context.flush_large(xtc3_context.has_large);
    }

    // Now it is time to compress all the data in the buffers with the bwlzh or base algo
    output_int(
        &mut output,
        &mut outdata,
        u32::try_from(xtc3_context.instructions.len()).expect("u32 from usize"),
    );
    if !xtc3_context.instructions.is_empty() {
        let mut bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.instructions.len())];
        let bwlzh_buf_len = if *speed >= 5 {
            bwlzh_compress(
                &xtc3_context.instructions,
                xtc3_context.instructions.len(),
                &mut bwlzh_buf,
            )
        } else {
            bwlzh_compress_no_lz77(
                &xtc3_context.instructions,
                xtc3_context.instructions.len(),
                &mut bwlzh_buf,
            )
        };
        output_int(
            &mut output,
            &mut outdata,
            u32::try_from(bwlzh_buf_len).expect("u32 from usize"),
        );
        output[outdata..outdata + bwlzh_buf_len].copy_from_slice(&bwlzh_buf[..bwlzh_buf_len]);
        outdata += bwlzh_buf_len;
    }

    output_int(
        &mut output,
        &mut outdata,
        u32::try_from(xtc3_context.rle.len()).expect("u32 from usize"),
    );
    if !xtc3_context.rle.is_empty() {
        let mut bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.rle.len())];
        let bwlzh_buf_len = if *speed >= 5 {
            bwlzh_compress(&xtc3_context.rle, xtc3_context.rle.len(), &mut bwlzh_buf)
        } else {
            bwlzh_compress_no_lz77(&xtc3_context.rle, xtc3_context.rle.len(), &mut bwlzh_buf)
        };
        output_int(
            &mut output,
            &mut outdata,
            u32::try_from(bwlzh_buf_len).expect("u32 from usize"),
        );
        output[outdata..outdata + bwlzh_buf_len].copy_from_slice(&bwlzh_buf[..bwlzh_buf_len]);
        outdata += bwlzh_buf_len;
    }

    output_int(
        &mut output,
        &mut outdata,
        u32::try_from(xtc3_context.large_direct.len()).expect("u32 from usize"),
    );
    let mut bwlzh_buf;
    let mut bwlzh_buf_len;
    if !xtc3_context.large_direct.is_empty() {
        if *speed <= 2
            || (*speed <= 5
                && (!heuristic_bwlzh(&xtc3_context.large_direct, xtc3_context.large_direct.len())))
        {
            bwlzh_buf = vec![];
            bwlzh_buf_len = usize::try_from(i32::MAX).expect("usize from i32");
        } else {
            bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.large_direct.len())];
            bwlzh_buf_len = if *speed >= 5 {
                bwlzh_compress(
                    &xtc3_context.large_direct,
                    xtc3_context.large_direct.len(),
                    &mut bwlzh_buf,
                )
            } else {
                bwlzh_compress_no_lz77(
                    &xtc3_context.large_direct,
                    xtc3_context.large_direct.len(),
                    &mut bwlzh_buf,
                )
            };
        }
        // If this can be written using base compression we should do that
        let (base_buf, base_buf_len) =
            base_compress(&xtc3_context.large_direct, xtc3_context.large_direct.len());
        base_or_bwlzh_output(
            &mut outdata,
            &mut output,
            &bwlzh_buf,
            bwlzh_buf_len,
            &base_buf,
            base_buf_len,
        );
    }

    output_int(
        &mut output,
        &mut outdata,
        u32::try_from(xtc3_context.large_intra_delta.len()).expect("u32 from usize"),
    );
    if !xtc3_context.large_intra_delta.is_empty() {
        if (*speed <= 2)
            || ((*speed <= 5)
                && (!heuristic_bwlzh(
                    &xtc3_context.large_intra_delta,
                    xtc3_context.large_intra_delta.len(),
                )))
        {
            bwlzh_buf = vec![];
            bwlzh_buf_len = usize::try_from(i32::MAX).expect("usize from i32");
        } else {
            bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.large_intra_delta.len())];
            bwlzh_buf_len = if *speed >= 5 {
                bwlzh_compress(
                    &xtc3_context.large_intra_delta,
                    xtc3_context.large_intra_delta.len(),
                    &mut bwlzh_buf,
                )
            } else {
                bwlzh_compress_no_lz77(
                    &xtc3_context.large_intra_delta,
                    xtc3_context.large_intra_delta.len(),
                    &mut bwlzh_buf,
                )
            };
        }
        // If this can be written smaller using base compression we should do that
        let (base_buf, base_buf_len) = base_compress(
            &xtc3_context.large_intra_delta,
            xtc3_context.large_intra_delta.len(),
        );
        base_or_bwlzh_output(
            &mut outdata,
            &mut output,
            &bwlzh_buf,
            bwlzh_buf_len,
            &base_buf,
            base_buf_len,
        );
    }

    output_int(
        &mut output,
        &mut outdata,
        u32::try_from(xtc3_context.large_inter_delta.len()).expect("u32 from usize"),
    );
    if !xtc3_context.large_inter_delta.is_empty() {
        if (*speed <= 2)
            || ((*speed <= 5)
                && (!heuristic_bwlzh(
                    &xtc3_context.large_inter_delta,
                    xtc3_context.large_inter_delta.len(),
                )))
        {
            bwlzh_buf = vec![];
            bwlzh_buf_len = usize::try_from(i32::MAX).expect("usize from i32");
        } else {
            bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.large_inter_delta.len())];
            bwlzh_buf_len = if *speed >= 5 {
                bwlzh_compress(
                    &xtc3_context.large_inter_delta,
                    xtc3_context.large_inter_delta.len(),
                    &mut bwlzh_buf,
                )
            } else {
                bwlzh_compress_no_lz77(
                    &xtc3_context.large_inter_delta,
                    xtc3_context.large_inter_delta.len(),
                    &mut bwlzh_buf,
                )
            };
        }
        // If this can be written smaller using base compression we should do that
        let (base_buf, base_buf_len) = base_compress(
            &xtc3_context.large_inter_delta,
            xtc3_context.large_inter_delta.len(),
        );
        base_or_bwlzh_output(
            &mut outdata,
            &mut output,
            &bwlzh_buf,
            bwlzh_buf_len,
            &base_buf,
            base_buf_len,
        );
    }

    output_int(
        &mut output,
        &mut outdata,
        u32::try_from(xtc3_context.smallintra.len()).expect("u32 from usize"),
    );
    if !xtc3_context.smallintra.is_empty() {
        if (*speed <= 2)
            || ((*speed <= 5)
                && (!heuristic_bwlzh(&xtc3_context.smallintra, xtc3_context.smallintra.len())))
        {
            bwlzh_buf = vec![];
            bwlzh_buf_len = usize::try_from(i32::MAX).expect("usize from i32");
        } else {
            bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.smallintra.len())];
            bwlzh_buf_len = if *speed >= 5 {
                bwlzh_compress(
                    &xtc3_context.smallintra,
                    xtc3_context.smallintra.len(),
                    &mut bwlzh_buf,
                )
            } else {
                bwlzh_compress_no_lz77(
                    &xtc3_context.smallintra,
                    xtc3_context.smallintra.len(),
                    &mut bwlzh_buf,
                )
            };
        }
        // If this can be written smaller using base compression we should do that
        let (base_buf, base_buf_len) =
            base_compress(&xtc3_context.smallintra, xtc3_context.smallintra.len());
        base_or_bwlzh_output(
            &mut outdata,
            &mut output,
            &bwlzh_buf,
            bwlzh_buf_len,
            &base_buf,
            base_buf_len,
        );
    }
    (output, outdata)
}

fn decompress_bwlzh_block(ptr: &mut &[u8], nvals: usize, vals: &mut Vec<u32>) {
    let bwlzh_buf_len = u32::from_le_bytes(ptr[..4].try_into().expect("error handling")) as usize;
    *ptr = &ptr[4..];
    vals.resize(nvals, 0);
    bwlzh_decompress(ptr, nvals as i32, vals);
    *ptr = &ptr[bwlzh_buf_len..];
}

fn decompress_base_block(ptr: &mut &[u8], nvals: usize, vals: &mut Vec<u32>) {
    let base_buf_len = u32::from_le_bytes(ptr[..4].try_into().expect("error handling")) as usize;
    *ptr = &ptr[4..];
    vals.resize(nvals, 0);
    base_decompress(&ptr[..base_buf_len], nvals, vals);
    *ptr = &ptr[base_buf_len..];
}

fn base_decompress(input: &[u8], len: usize, output: &mut [u32]) {
    let maxbasevals = (u16::from(input[0]) | (u16::from(input[1]) << 8)) as usize;
    let baseinterval = input[2] as usize;
    let mut input = &input[3..];

    let mut largeint = vec![0u32; maxbasevals + 1];
    let mut largeint_tmp = vec![0u32; maxbasevals + 1];

    for ixyz in 0..3 {
        let mut nvals_left = len / 3;
        let mut outvals = ixyz;
        let mut basegiven = 0usize;
        let mut base = 0u32;
        let mut numbytes = 0usize;

        while nvals_left > 0 {
            if basegiven == 0 {
                base = u32::from_le_bytes(input[..4].try_into().expect("error handling"));
                input = &input[4..];
                basegiven = baseinterval;
                numbytes = base_bytes(base, maxbasevals);
            }
            basegiven -= 1;

            if nvals_left < maxbasevals {
                numbytes = base_bytes(base, nvals_left);
            }

            largeint.fill(0);

            if numbytes / 4 < maxbasevals + 1 {
                for (j, item) in input.iter().enumerate().take(numbytes) {
                    let ilarge = j / 4;
                    let ibyte = j % 4;
                    largeint[ilarge] |= u32::from(*item) << (ibyte * 8);
                }
            }
            input = &input[numbytes..];

            let n = maxbasevals.min(nvals_left);
            for i in (0..n).rev() {
                output[outvals + i * 3] =
                    ptngc_largeint_div(base, &largeint, &mut largeint_tmp, maxbasevals + 1);
                largeint[..maxbasevals + 1].copy_from_slice(&largeint_tmp[..maxbasevals + 1]);
            }
            outvals += n * 3;
            nvals_left -= n;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unpack_one_large(
    ctx: &Xtc3Context,
    ilargedir: &mut usize,
    ilargeintra: &mut usize,
    ilargeinter: &mut usize,
    prevcoord: &mut [i32; 3],
    minint: &[i32; 3],
    output: &mut [i32],
    outdata: usize,
    didswap: bool,
    natoms: usize,
    current_large_type: i32,
) {
    let mut large_ints = [0i32; 3];
    if current_large_type == 0 && !ctx.large_direct.is_empty() {
        large_ints[0] = ctx.large_direct[*ilargedir] as i32 + minint[0];
        large_ints[1] = ctx.large_direct[*ilargedir + 1] as i32 + minint[1];
        large_ints[2] = ctx.large_direct[*ilargedir + 2] as i32 + minint[2];
        *ilargedir += 3;
    } else if current_large_type == 1 && !ctx.large_intra_delta.is_empty() {
        large_ints[0] =
            unpositive_int(ctx.large_intra_delta[*ilargeintra].cast_signed()) + prevcoord[0];
        large_ints[1] =
            unpositive_int(ctx.large_intra_delta[*ilargeintra + 1].cast_signed()) + prevcoord[1];
        large_ints[2] =
            unpositive_int(ctx.large_intra_delta[*ilargeintra + 2].cast_signed()) + prevcoord[2];
        *ilargeintra += 3;
    } else if !ctx.large_inter_delta.is_empty() {
        let swap_offset = if didswap { 3 } else { 0 };
        let base = outdata - natoms * 3 + swap_offset;
        large_ints[0] = unpositive_int(ctx.large_inter_delta[*ilargeinter] as i32) + output[base];
        large_ints[1] =
            unpositive_int(ctx.large_inter_delta[*ilargeinter + 1] as i32) + output[base + 1];
        large_ints[2] =
            unpositive_int(ctx.large_inter_delta[*ilargeinter + 2] as i32) + output[base + 2];
        *ilargeinter += 3;
    }
    *prevcoord = large_ints;
    output[outdata] = large_ints[0];
    output[outdata + 1] = large_ints[1];
    output[outdata + 2] = large_ints[2];
}

pub(crate) fn ptngc_unpack_array_xtc3(
    packed: &[u8],
    output: &mut [i32],
    length: i32,
    n_atoms: usize,
) {
    let mut xtc3_context = Xtc3Context::default();
    let mut ptr = packed;
    let mut minint = [0i32; 3];

    for item in &mut minint {
        *item = unpositive_int(i32::from_le_bytes(
            ptr[..4].try_into().expect("error handling"),
        ));
        ptr = &ptr[4..];
    }

    xtc3_context.ninstr = i32::from_le_bytes(ptr[..4].try_into().expect("error handling"));
    ptr = &ptr[4..];
    if xtc3_context.ninstr != 0 {
        decompress_bwlzh_block(
            &mut ptr,
            xtc3_context.ninstr as usize,
            &mut xtc3_context.instructions,
        );
    }

    let nrle = i32::from_le_bytes(ptr[..4].try_into().expect("error handling"));
    ptr = &ptr[4..];
    if nrle != 0 {
        decompress_bwlzh_block(&mut ptr, nrle as usize, &mut xtc3_context.rle);
    }

    let nlargedir = i32::from_le_bytes(ptr[..4].try_into().expect("error handling"));
    ptr = &ptr[4..];
    if nlargedir != 0 {
        let use_bwlzh = ptr[0] == 1;
        ptr = &ptr[1..];
        if use_bwlzh {
            decompress_bwlzh_block(&mut ptr, nlargedir as usize, &mut xtc3_context.large_direct);
        } else {
            decompress_base_block(&mut ptr, nlargedir as usize, &mut xtc3_context.large_direct);
        }
    }

    let nlargeintra = i32::from_le_bytes(ptr[..4].try_into().expect("error handling"));
    ptr = &ptr[4..];
    if nlargeintra != 0 {
        let use_bwlzh = ptr[0] == 1;
        ptr = &ptr[1..];
        if use_bwlzh {
            decompress_bwlzh_block(
                &mut ptr,
                nlargeintra as usize,
                &mut xtc3_context.large_intra_delta,
            );
        } else {
            decompress_base_block(
                &mut ptr,
                nlargeintra as usize,
                &mut xtc3_context.large_intra_delta,
            );
        }
    }

    let nlargeinter = i32::from_le_bytes(ptr[..4].try_into().expect("error handling"));
    ptr = &ptr[4..];
    if nlargeinter != 0 {
        let use_bwlzh = ptr[0] == 1;
        ptr = &ptr[1..];
        if use_bwlzh {
            decompress_bwlzh_block(
                &mut ptr,
                nlargeinter as usize,
                &mut xtc3_context.large_inter_delta,
            );
        } else {
            decompress_base_block(
                &mut ptr,
                nlargeinter as usize,
                &mut xtc3_context.large_inter_delta,
            );
        }
    }

    let nsmallintra = i32::from_le_bytes(ptr[..4].try_into().expect("error handling"));
    ptr = &ptr[4..];
    if nsmallintra != 0 {
        let use_bwlzh = ptr[0] == 1;
        ptr = &ptr[1..];
        if use_bwlzh {
            decompress_bwlzh_block(&mut ptr, nsmallintra as usize, &mut xtc3_context.smallintra);
        } else {
            decompress_base_block(&mut ptr, nsmallintra as usize, &mut xtc3_context.smallintra);
        }
    }

    // Initial prevcoord is the minimum integers.
    let mut prevcoord = [minint[0], minint[1], minint[2]];
    let mut ntriplets_left = length / 3;
    let mut outdata = 0usize;
    let mut swapatoms = false;
    let mut runlength = 0usize;
    let mut current_large_type = 0i32;
    let mut iinstr = 0usize;
    let mut irle = 0usize;
    let mut ilargedir = 0usize;
    let mut ilargeintra = 0usize;
    let mut ilargeinter = 0usize;
    let mut ismallintra = 0usize;

    while ntriplets_left > 0 && iinstr < xtc3_context.ninstr as usize {
        let instr = xtc3_context.instructions[iinstr];
        iinstr += 1;

        if instr == INSTR_DEFAULT || instr == INSTR_ONLY_LARGE || instr == INSTR_ONLY_SMALL {
            if instr != INSTR_ONLY_SMALL {
                let didswap = instr == INSTR_DEFAULT && swapatoms;
                unpack_one_large(
                    &xtc3_context,
                    &mut ilargedir,
                    &mut ilargeintra,
                    &mut ilargeinter,
                    &mut prevcoord,
                    &minint,
                    output,
                    outdata,
                    didswap,
                    n_atoms,
                    current_large_type,
                );
                ntriplets_left -= 1;
                outdata += 3;
            }
            if instr != INSTR_ONLY_LARGE {
                for i in 0..runlength {
                    prevcoord[0] +=
                        unpositive_int(xtc3_context.smallintra[ismallintra].cast_signed());
                    prevcoord[1] +=
                        unpositive_int(xtc3_context.smallintra[ismallintra + 1].cast_signed());
                    prevcoord[2] +=
                        unpositive_int(xtc3_context.smallintra[ismallintra + 2].cast_signed());
                    ismallintra += 3;
                    output[outdata + i * 3] = prevcoord[0];
                    output[outdata + i * 3 + 1] = prevcoord[1];
                    output[outdata + i * 3 + 2] = prevcoord[2];
                }
                if instr == INSTR_DEFAULT && swapatoms {
                    for i in 0..3 {
                        output.swap(outdata - 3 + i, outdata + i);
                    }
                }
                ntriplets_left -= runlength as i32;
                outdata += runlength * 3;
            }
        } else if instr == INSTR_LARGE_RLE && irle < xtc3_context.rle.len() {
            let large_rle = xtc3_context.rle[irle] as usize;
            irle += 1;
            for _ in 0..large_rle {
                unpack_one_large(
                    &xtc3_context,
                    &mut ilargedir,
                    &mut ilargeintra,
                    &mut ilargeinter,
                    &mut prevcoord,
                    &minint,
                    output,
                    outdata,
                    false,
                    n_atoms,
                    current_large_type,
                );
                ntriplets_left -= 1;
                outdata += 3;
            }
        } else if instr == INSTR_SMALL_RUNLENGTH && irle < xtc3_context.rle.len() {
            runlength = xtc3_context.rle[irle] as usize;
            irle += 1;
        } else if instr == INSTR_FLIP {
            swapatoms = !swapatoms;
        } else if instr == INSTR_LARGE_DIRECT {
            current_large_type = 0;
        } else if instr == INSTR_LARGE_INTRA_DELTA {
            current_large_type = 1;
        } else if instr == INSTR_LARGE_INTER_DELTA {
            current_large_type = 2;
        }
    }
}

fn base_or_bwlzh_output(
    outdata: &mut usize,
    output: &mut [u8],
    bwlzh_buf: &[u8],
    bwlzh_buf_len: usize,
    base_buf: &[u8],
    base_buf_len: usize,
) {
    if base_buf_len < bwlzh_buf_len {
        output[*outdata] = 0;
        *outdata += 1;
        output_int(
            output,
            outdata,
            u32::try_from(base_buf_len).expect("u32 from usize"),
        );
        output[*outdata..*outdata + base_buf_len].copy_from_slice(base_buf);
        *outdata += base_buf_len;
    } else {
        output[*outdata] = 1;
        *outdata += 1;
        output_int(
            output,
            outdata,
            u32::try_from(bwlzh_buf_len).expect("u32 from usize"),
        );
        output[*outdata..*outdata + bwlzh_buf_len].copy_from_slice(&bwlzh_buf[..bwlzh_buf_len]);
        *outdata += bwlzh_buf_len;
    }
}

fn heuristic_bwlzh(ints: &[u32], nints: usize) -> bool {
    let num = ints[..nints]
        .iter()
        .filter(|&&v| v >= 16384)
        .fold(0, |acc, _| acc + 1);
    num <= nints / 10
}

#[cfg(test)]
mod compress_tests {
    use super::*;

    #[test]
    fn empty_input() {
        let data = [];
        let (output, _) = base_compress(&data, 0);
        assert_eq!(output, [24, 0, 8]);
    }

    #[test]
    fn single_zero() {
        let data = [0];
        let (output, _) = base_compress(&data, 1);
        assert_eq!(output, [24, 0, 8, 2, 0, 0, 0, 0]);
    }

    #[test]
    fn single_nonzero() {
        let data = [12345];
        let (output, _) = base_compress(&data, 1);
        assert_eq!(output, [24, 0, 8, 58, 48, 0, 0, 57, 48]);
    }

    #[test]
    fn block_of_maxbasevals() {
        let data = (0..MAXBASEVALS as u32).collect::<Vec<_>>();
        let (output, _) = base_compress(&data, MAXBASEVALS);
        assert_eq!(
            output,
            [
                24, 0, 8, 22, 0, 0, 0, 173, 47, 64, 22, 0, 23, 0, 0, 0, 236, 161, 25, 241, 0, 24,
                0, 0, 0, 55, 204, 186, 95, 2
            ]
        );
    }

    #[test]
    fn repeat_value() {
        let data = [7; 32];
        let (output, _) = base_compress(&data, 32);
        assert_eq!(
            output,
            [
                24, 0, 8, 8, 0, 0, 0, 255, 255, 255, 255, 1, 8, 0, 0, 0, 255, 255, 255, 255, 1, 8,
                0, 0, 0, 255, 255, 255, 63
            ]
        );
    }

    #[test]
    fn squares() {
        let data = (0..20).map(|v| v * v).collect::<Vec<_>>();
        let (output, _) = base_compress(&data, 20);
        assert_eq!(
            output,
            [
                24, 0, 8, 69, 1, 0, 0, 71, 20, 239, 42, 12, 30, 0, 0, 106, 1, 0, 0, 221, 107, 73,
                51, 235, 89, 8, 0, 34, 1, 0, 0, 157, 213, 218, 200, 159, 7, 0
            ]
        );
    }
}

#[cfg(test)]
mod pack_tests {
    use super::*;

    #[test]
    fn one_atom() {
        let mut input = vec![100, 200, 300];
        let natoms = 1;
        let mut speed = 1;
        let mut length = input.len();
        let (output, output_len) =
            ptngc_pack_array_xtc3(&mut input, &mut length, natoms, &mut speed);
        assert_eq!(
            output[..output_len],
            [
                199, 0, 0, 0, 143, 1, 0, 0, 87, 2, 0, 0, 1, 0, 0, 0, 119, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 26, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0, 4, 0, 0, 8, 64, 0, 1, 0, 0, 0, 25, 0, 0,
                0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 2, 0, 0, 33, 0,
                1, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1,
                0, 0, 2, 0, 0, 33, 0, 0, 0, 0, 3, 0, 0, 0, 0, 18, 0, 0, 0, 24, 0, 8, 2, 0, 0, 0, 0,
                2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]
        );
    }

    #[test]
    fn three_atoms() {
        #[rustfmt::skip]
        let mut input = vec![
            100, 200, 300,
            101, 201, 301,
            99, 199, 299,
        ];
        let natoms = 3;
        let mut speed = 3;
        let mut length = input.len();
        let (output, output_len) =
            ptngc_pack_array_xtc3(&mut input, &mut length, natoms, &mut speed);
        assert_eq!(
            output[..output_len],
            [
                197, 0, 0, 0, 141, 1, 0, 0, 85, 2, 0, 0, 5, 0, 0, 0, 127, 0, 0, 0, 5, 0, 0, 0, 5,
                0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 5, 0, 0, 0, 30, 0, 0, 0, 1, 0, 5, 0, 0, 0, 5,
                0, 0, 0, 2, 0, 0, 0, 222, 16, 8, 0, 0, 5, 0, 0, 9, 0, 0, 137, 16, 137, 28, 96, 0,
                3, 0, 0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 152, 6, 0, 0, 3,
                0, 0, 2, 0, 0, 134, 40, 128, 0, 3, 0, 0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0,
                0, 1, 0, 0, 0, 152, 6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128, 0, 0, 0, 0, 6, 0, 0,
                0, 0, 18, 0, 0, 0, 24, 0, 8, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 3, 0, 0,
                0, 0, 18, 0, 0, 0, 24, 0, 8, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0
            ]
        );
    }

    #[test]
    fn sequential_coordinates() {
        #[rustfmt::skip]
        let mut input = vec![
            0,   0,   0,    // atom 0
            10,  10,  10,   // atom 1
            20,  20,  20,   // atom 2
            30,  30,  30,   // atom 3
            40,  40,  40,   // atom 4
            50,  50,  50    // atom 5
        ];
        let natoms = 6;
        let mut speed = 4;
        let mut length = input.len();
        let (output, output_len) =
            ptngc_pack_array_xtc3(&mut input, &mut length, natoms, &mut speed);
        assert_eq!(
            output[..output_len],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 127, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0,
                0, 5, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 30, 0, 0, 0, 1, 0, 5, 0, 0, 0, 5, 0, 0,
                0, 2, 0, 0, 0, 240, 224, 8, 0, 0, 5, 0, 0, 9, 0, 0, 137, 17, 17, 28, 96, 0, 3, 0,
                0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 152, 6, 0, 0, 3, 0, 0,
                2, 0, 0, 134, 40, 128, 0, 3, 0, 0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1,
                0, 0, 0, 152, 6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128, 1, 0, 0, 0, 119, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 26, 0, 0, 0, 1, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0, 5, 0, 0, 4, 32, 0, 1, 0, 0,
                0, 25, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 2,
                0, 0, 33, 0, 1, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                4, 0, 0, 1, 0, 0, 2, 0, 0, 33, 9, 0, 0, 0, 0, 21, 0, 0, 0, 24, 0, 8, 21, 0, 0, 0,
                230, 0, 21, 0, 0, 0, 230, 0, 21, 0, 0, 0, 230, 0, 9, 0, 0, 0, 0, 21, 0, 0, 0, 24,
                0, 8, 20, 0, 0, 0, 63, 31, 20, 0, 0, 0, 63, 31, 20, 0, 0, 0, 63, 31, 0, 0, 0, 0, 0,
                0, 0, 0,
            ]
        );
    }

    #[test]
    fn speed_variations() {
        // this gives the same output no matter the speed - not sure if that is expected?
        #[rustfmt::skip]
        let mut input = vec![
            500, 1000, 1500,   // atom 0
            502, 1002, 1502,   // atom 1
            498,  998, 1498,   // atom 2
            504, 1004, 1504    // atom 3
        ];
        let natoms = 4;
        let mut length = input.len();
        let expected_outputs = [[
            227, 3, 0, 0, 203, 7, 0, 0, 179, 11, 0, 0, 2, 0, 0, 0, 122, 0, 0, 0, 2, 0, 0, 0, 2, 0,
            0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 27, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0,
            1, 0, 0, 0, 128, 6, 0, 0, 2, 0, 0, 7, 0, 0, 4, 40, 64, 0, 2, 0, 0, 0, 26, 0, 0, 0, 1,
            0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 64, 5, 0, 0, 2, 0, 0, 2, 0, 0, 133, 8, 0, 2, 0,
            0, 0, 26, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 64, 5, 0, 0, 2, 0, 0, 2,
            0, 0, 133, 8, 1, 0, 0, 0, 119, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 26, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 1, 0,
            0, 5, 0, 0, 4, 32, 0, 1, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 4, 0, 0, 1, 0, 0, 2, 0, 0, 33, 0, 1, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 2, 0, 0, 33, 12, 0, 0, 0, 0, 21, 0, 0, 0, 24,
            0, 8, 7, 0, 0, 0, 120, 3, 7, 0, 0, 0, 120, 3, 7, 0, 0, 0, 120, 3, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ]; 6];

        for (speed, expected) in (1..7).zip(expected_outputs) {
            let mut tmp_speed = speed;
            let (output, output_len) =
                ptngc_pack_array_xtc3(&mut input, &mut length, natoms, &mut tmp_speed);
            assert_eq!(output[..output_len], expected,);
        }
    }

    #[test]
    fn zero_coordinates() {
        #[rustfmt::skip]
        let mut input = vec![
            0,   0,   0,    // atom 0
            10,  10,  10,   // atom 1
            0,   0,   0,   // atom 2
            30,  30,  30,   // atom 3
        ];
        let natoms = 4;
        let mut speed = 6;
        let mut length = input.len();
        let (output, output_len) =
            ptngc_pack_array_xtc3(&mut input, &mut length, natoms, &mut speed);
        assert_eq!(
            output[..output_len],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 122, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0,
                0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 27, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0,
                0, 1, 0, 0, 0, 128, 6, 0, 0, 2, 0, 0, 7, 0, 0, 4, 40, 64, 0, 2, 0, 0, 0, 26, 0, 0,
                0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 64, 5, 0, 0, 2, 0, 0, 2, 0, 0, 133, 8,
                0, 2, 0, 0, 0, 26, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 64, 5, 0, 0,
                2, 0, 0, 2, 0, 0, 133, 8, 1, 0, 0, 0, 119, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 26, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 5, 0, 0, 1, 0, 0, 5, 0, 0, 4, 32, 0, 1, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 2, 0, 0, 33, 0, 1, 0, 0, 0, 25,
                0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 2, 0, 0,
                33, 12, 0, 0, 0, 0, 24, 0, 0, 0, 24, 0, 8, 31, 0, 0, 0, 168, 37, 0, 31, 0, 0, 0,
                168, 37, 0, 31, 0, 0, 0, 168, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]
        );
    }
}

#[cfg(test)]
mod int_tests {
    use super::*;

    #[test]
    fn zero() {
        assert_eq!(positive_int(0), 0);
    }

    #[test]
    fn positive_numbers() {
        assert_eq!(positive_int(1), 1);
        assert_eq!(positive_int(2), 3);
        assert_eq!(positive_int(3), 5);
        assert_eq!(positive_int(4), 7);
        assert_eq!(positive_int(5), 9);
        assert_eq!(positive_int(10), 19);
        assert_eq!(positive_int(100), 199);
    }

    #[test]
    fn negative_numbers() {
        assert_eq!(positive_int(-1), 2);
        assert_eq!(positive_int(-2), 4);
        assert_eq!(positive_int(-3), 6);
        assert_eq!(positive_int(-4), 8);
        assert_eq!(positive_int(-5), 10);
        assert_eq!(positive_int(-10), 20);
        assert_eq!(positive_int(-100), 200);
    }

    #[test]
    fn weird_cases() {
        // Test some larger values
        assert_eq!(positive_int(1000), 1999);
        assert_eq!(positive_int(-1000), 2000);

        // Test boundary values
        assert_eq!(positive_int(i32::MAX), 4294967293);
    }
}
