use crate::{
    bwlzh::{bwlzh_compress, bwlzh_compress_gen, bwlzh_compress_no_lz77, bwlzh_get_buflen},
    utils::copy_bytes,
    widemuldiv::{ptngc_largeint_add, ptngc_largeint_mul},
    xtc2::{ptngc_find_magic_index, ptngc_magic},
};

const IFLIPGAINCHECK: f64 = 0.89089871814033927; /*  1./(2**(1./6)) */

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
const QUITE_LARGE: usize = 3;
const IS_LARGE: usize = 6;

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
    ninstr: usize,
    ninstr_alloc: usize,
    rle: Vec<u32>,
    nrle: usize,
    nrle_alloc: usize,
    large_direct: Vec<u32>,
    nlargedir: usize,
    nlargedir_alloc: usize,
    large_intra_delta: Vec<u32>,
    nlargeintra: usize,
    nlargeintra_alloc: usize,
    large_inter_delta: Vec<u32>,
    nlargeinter: usize,
    nlargeinter_alloc: usize,
    smallintra: Vec<u32>,
    nsmallintra: usize,
    nsmallintra_alloc: usize,
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
            ninstr_alloc: 0,
            rle: Vec::new(),
            nrle: 0,
            nrle_alloc: 0,
            large_direct: Vec::new(),
            nlargedir: 0,
            nlargedir_alloc: 0,
            large_intra_delta: Vec::new(),
            nlargeintra: 0,
            nlargeintra_alloc: 0,
            large_inter_delta: Vec::new(),
            nlargeinter: 0,
            nlargeinter_alloc: 0,
            smallintra: Vec::new(),
            nsmallintra: 0,
            nsmallintra_alloc: 0,
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
fn swap_ints(input: &[i32], output: &mut [i32]) {
    output[0] = input[0] + input[1];
    output[1] = -input[1];
    output[2] = input[1] + input[2];
}

fn swap_is_better(input: &[i32], minint: &[i32; 3]) -> (u32, u32) {
    let mut normal_max = 0;
    let mut swapped_max = 0;
    let mut normal = [0; 3];
    let mut swapped = [0; 3];

    for i in 0..3 {
        normal[0] = input[i] - minint[i];
        normal[1] = input[3 + i] - input[i]; // minint[i]-minint[i] cancels out
        normal[2] = input[6 + i] - input[3 + i]; // minint[i]-minint[i] cancels out
        swap_ints(&mut normal, &mut swapped);

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
            } else {
                if *swapatoms {
                    *swapatoms = false;
                    didswap = true;
                }
            }
        }

        if didswap {
            self.instructions.push(INSTR_FLIP);
        }
    }

    fn write_three_large(&mut self, i: usize) {
        match self.current_large_type {
            0 => {
                for m in 0..3 {
                    self.large_direct.push(self.has_large_ints[i * 3 + m]);
                }
            }
            1 => {
                for m in 0..3 {
                    self.large_intra_delta.push(self.has_large_ints[i * 3 + m]);
                }
            }
            _ => {
                for m in 0..3 {
                    self.large_inter_delta.push(self.has_large_ints[i * 3 + m]);
                }
            }
        }
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
                self.instructions.push(j.try_into().expect("usize to u32"));
                for k in 0..j {
                    self.write_three_large(i + k);
                }
            }
            i += j;
        }
        if self.has_large - n != 0 {
            let j = 0;
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
            (input[inpdata] - self.minint[0])
                .try_into()
                .expect("i32 to u32"),
            (input[inpdata + 1] - self.minint[1])
                .try_into()
                .expect("i32 to u32"),
            (input[inpdata + 2] - self.minint[2])
                .try_into()
                .expect("i32 to u32"),
        ];
        let mut minlen = compute_intlen(direct.as_slice());
        let mut best_type = 0; // Direct

        // in c code: #if 1
        // Then try intra coding if we can
        if intradelta_ok && atomframe >= 3 {
            intradelta[0] = positive_int(input[inpdata] - input[inpdata - 3]);
            intradelta[1] = positive_int(input[inpdata + 1] - input[inpdata - 2]);
            intradelta[2] = positive_int(input[inpdata + 2] - input[inpdata - 1]);
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
            interdelta[0] = positive_int(input[inpdata] - input[inpdata - natoms * 3]);
            interdelta[1] = positive_int(input[inpdata + 1] - input[inpdata - natoms * 2 + 1]);
            interdelta[2] = positive_int(input[inpdata + 2] - input[inpdata - natoms * 1 + 2]);
            let thislen = compute_intlen(interdelta.as_slice());
            if thislen * THRESHOLD_INTRA_INTER_DIRECT < minlen {
                best_type = 2; // Inter delta
            }
        }
        // in c code: #endif
        self.has_large_type[self.has_large] = best_type;
        match best_type {
            0 => {
                self.has_large_ints[self.has_large * 3] = direct[0];
                self.has_large_ints[self.has_large * 3 + 1] = direct[1];
                self.has_large_ints[self.has_large * 3 + 2] = direct[2];
            }
            1 => {
                self.has_large_ints[self.has_large * 3] = intradelta[0];
                self.has_large_ints[self.has_large * 3 + 1] = intradelta[1];
                self.has_large_ints[self.has_large * 3 + 2] = intradelta[2];
            }
            2 => {
                self.has_large_ints[self.has_large * 3] = interdelta[0];
                self.has_large_ints[self.has_large * 3 + 1] = interdelta[1];
                self.has_large_ints[self.has_large * 3 + 2] = interdelta[2];
            }
            _ => panic!("unknown best_type: {best_type}"),
        }
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

fn positive_int(item: i32) -> u32 {
    match item {
        i if i > 0 => 1 + (u32::try_from(i).expect("u32 from i32") - 1) * 2,
        i if i < 0 => 2 + (u32::try_from(-i).expect("u32 from i32") - 1) * 2,
        _ => 0, // Case when item == 0
    }
}

fn unpositive_int(val: i32) -> i32 {
    let mut s = (val + 1) / 2;
    if val % 2 == 0 {
        s = -s;
    }
    s
}

fn compute_intlen(ints: &[u32]) -> f64 {
    /* The largest value. */
    let mut m = ints[0];
    if ints[1] > m {
        m = ints[1];
    }
    if ints[2] > m {
        m = ints[2];
    }
    m.into()
}

fn insert_batch(
    input_ptr: &[i32],
    ntriplets_left: usize,
    prevcoord: &[i32],
    encode_ints: &mut [i32],
    startenc: usize,
) -> usize {
    let mut nencode = startenc * 3;
    let mut tmp_prevcoord = [prevcoord[0], prevcoord[1], prevcoord[2]];

    if startenc > 0 {
        for i in 0..startenc {
            tmp_prevcoord[0] += encode_ints[i * 3];
            tmp_prevcoord[1] += encode_ints[i * 3 + 1];
            tmp_prevcoord[2] += encode_ints[i * 3 + 2];
        }
    }

    while (nencode < 3 + MAX_SMALL_RLE * 3) && (nencode < ntriplets_left * 3) {
        encode_ints[nencode] = input_ptr[nencode] - tmp_prevcoord[0];
        encode_ints[nencode + 1] = input_ptr[nencode + 1] - tmp_prevcoord[1];
        encode_ints[nencode + 2] = input_ptr[nencode + 2] - tmp_prevcoord[2];

        tmp_prevcoord[0] = input_ptr[nencode];
        tmp_prevcoord[1] = input_ptr[nencode + 1];
        tmp_prevcoord[2] = input_ptr[nencode + 2];
        nencode += 3;
    }
    nencode
}

// It is "large" if we have to increase the small index quite a
// bit. Not so much to be rejected by the not very large check
// later.
fn is_quite_large(input: &[i32], small_index: usize, max_large_index: usize) -> bool {
    let mut is = false;
    if small_index + QUITE_LARGE >= max_large_index {
        is = true
    } else {
        for inp in input.iter().take(3) {
            if positive_int(*inp) > ptngc_magic(small_index + QUITE_LARGE) {
                is = true;
                break;
            }
        }
    }
    is
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
) {
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
        (xtc3_context.maxint[0] - xtc3_context.minint[0])
            .try_into()
            .expect("i32 to u32"),
    );
    large_index[1] = ptngc_find_magic_index(
        (xtc3_context.maxint[1] - xtc3_context.minint[1])
            .try_into()
            .expect("i32 to u32"),
    );
    large_index[2] = ptngc_find_magic_index(
        (xtc3_context.maxint[2] - xtc3_context.minint[2])
            .try_into()
            .expect("i32 to u32"),
    );
    let max_large_index = *large_index.iter().max().expect("large_index to be init");

    // Guess initial small index
    let mut small_index = max_large_index / 2;

    // Find the largest value that is not large. Not large is half index of large
    let max_small = ptngc_magic(small_index);
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

    copy_bytes(
        positive_int(xtc3_context.minint[0]),
        &mut output,
        &mut outdata,
    );
    copy_bytes(
        positive_int(xtc3_context.minint[1]),
        &mut output,
        &mut outdata,
    );
    copy_bytes(
        positive_int(xtc3_context.minint[2]),
        &mut output,
        &mut outdata,
    );

    // Initial prevcoord is the minimum integers
    let mut prevcoord = [
        xtc3_context.minint[0],
        xtc3_context.minint[1],
        xtc3_context.minint[2],
    ];

    while ntriplets_left > 0 {
        // if ntriplets_left < 0 {
        //     eprintln!("TRAJNG: BUG! ntriplets_left < 0!");
        // }

        // If only less than three atoms left we just write them all as large integers. Here no swapping is done!
        if ntriplets_left < 3 {
            // TODO: does this loop not only run once?
            xtc3_context.buffer_large(input, inpdata, natoms, true);
            inpdata += 3;
            ntriplets_left -= 1;
            xtc3_context.flush_large(xtc3_context.has_large); // Flush all
        } else {
            let mut min_runlength = 0;
            let mut largest_required_base;
            let mut largest_runlength_base;
            let mut largest_required_index = 0;
            let mut largest_runlength_index = 0;
            let mut new_runlength = 0;
            let mut new_small_index = 0;
            let mut iter_runlength = 0;
            let mut iter_small_index = 0;
            let mut rle_index_dep = 0;

            didswap = false;
            // Insert the next batch of integers to be encoded into the buffer
            let mut nencode = insert_batch(
                &input,
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
                            let mut input = [0; 3];
                            let mut output = [0; 3];
                            input[0] = input[inpdata + i];
                            input[1] = input[inpdata + 3 + i] - input[inpdata + i];
                            input[2] = input[inpdata + 6 + i] - input[inpdata + 3 + i];
                            swap_ints(&input, &mut output);
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
                    for ienc in 0..3 {
                        prevcoord[ienc] = input[inpdata + ienc];
                    }
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
            for ienc in 0..nencode {
                encode_ints[ienc] = positive_int(encode_ints[ienc])
                    .try_into()
                    .expect("i32 from u32");
            }
            // Now we must decide what base and runlength to do. If we have swapped atoms it will be
            // at least 2. If even the next atom is large, we will not do anything
            largest_required_base = 0;

            // Determine required base
            largest_runlength_base = (0..(min_runlength * 3))
                .filter_map(|i| encode_ints.get(i))
                .filter(|&&val| val > largest_required_base)
                .next_back()
                .copied()
                .unwrap_or(0);
            // Also compute what the largest base is for the current runlength setting!
            // largest_runlength_base = 0;
            // for ienc in 0..(runlength * 3).min(nencode) {
            //     if encode_ints[ienc] > largest_runlength_base {
            //         largest_runlength_base = encode_ints[ienc];
            //     }
            // }

            largest_runlength_base = (0..(min_runlength * 3).min(nencode))
                .filter_map(|i| encode_ints.get(i))
                .filter(|&&val| val > largest_runlength_base)
                .next_back()
                .copied()
                .unwrap_or(0);

            largest_required_index =
                ptngc_find_magic_index(largest_required_base.try_into().expect("i32 to u32"));
            largest_runlength_index =
                ptngc_find_magic_index(largest_runlength_base.try_into().expect("i32 to u32"));

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
                    let test_index =
                        ptngc_find_magic_index(encode_ints[ienc].try_into().expect("i32 to u32"));
                    if test_index > new_small_index {
                        break;
                    }

                    ienc += 1;
                }
                if ienc / 3 > new_runlength {
                    iter_runlength = ienc / 3;
                }

                // How large encoding do we have to use?
                largest_runlength_base = (0..iter_runlength * 3)
                    .filter_map(|i| encode_ints.get(i))
                    .filter(|&&val| val > largest_runlength_base)
                    .next_back()
                    .copied()
                    .unwrap_or(0);
                largest_runlength_index = ptngc_find_magic_index(
                    largest_runlength_base.try_into().expect("u32 from i32"),
                );
                if largest_runlength_index != new_small_index {
                    iter_small_index = largest_runlength_index;
                }

                // to emulate the do .. while construct in c, we put this if statement at the end
                if !((new_runlength != iter_runlength) || (new_small_index != iter_small_index)) {
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
                        let mut delta = [0; 3];
                        let mut delta2 = [0; 3];
                        delta[0] = positive_int(
                            input[inpdata + i * 3] - input[inpdata - natoms * 3 + i * 3],
                        );
                        delta[1] = positive_int(
                            input[inpdata + i * 3 + 1] - input[inpdata - natoms * 3 + i * 3 + 1],
                        );
                        delta[2] = positive_int(
                            input[inpdata + i * 3 + 2] - input[inpdata - natoms * 3 + i * 3 + 2],
                        );
                        delta2[0] = positive_int(encode_ints[i * 3]);
                        delta2[1] = positive_int(encode_ints[i * 3 + 1]);
                        delta2[2] = positive_int(encode_ints[i * 3 + 2]);
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
                } else if new_runlength != runlength || new_small_index != small_index {
                    let mut change: i32 =
                        i32::try_from(new_small_index - small_index).expect("i32 from usize");
                    if new_small_index <= 0 {
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

                                let change_usize = usize::try_from(change).expect("usize from i32");
                                if isum
                                    > f64::from(ptngc_magic(small_index + change_usize))
                                        * f64::from(ptngc_magic(small_index + change_usize))
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

                    // Always accep the new small indices here
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
                for &item in encode_ints[..runlength * 3].iter() {
                    xtc3_context
                        .smallintra
                        .push(item.try_into().expect("i32 to u32"));
                }
                // Update `prevcoord`
                for ienc in 0..runlength {
                    prevcoord[0] += unpositive_int(encode_ints[ienc * 3]);
                    prevcoord[1] += unpositive_int(encode_ints[ienc * 3 + 1]);
                    prevcoord[2] += unpositive_int(encode_ints[ienc * 3 + 2]);
                }
                inpdata += 3;
                ntriplets_left -= runlength;
            } else {
                refused += 1;
            }
        }

        // If we have large previous integers we must flush them now
        if xtc3_context.has_large > 0 {
            xtc3_context.flush_large(xtc3_context.has_large);
        }

        // Now it is time to compress all the data in the buffers with the bwlzh or base algo
        // in c code: output_int
        let bytes = xtc3_context.ninstr.to_ne_bytes();
        output[outdata..outdata + 4].copy_from_slice(&bytes);
        outdata += 4;

        if xtc3_context.ninstr > 0 {
            let mut bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.ninstr)];
            let bwlzh_buf_len = if *speed >= 5 {
                bwlzh_compress(
                    &xtc3_context.instructions,
                    xtc3_context.ninstr,
                    &mut bwlzh_buf,
                )
            } else {
                bwlzh_compress_no_lz77(
                    &xtc3_context.instructions,
                    xtc3_context.ninstr,
                    &mut bwlzh_buf,
                )
            };
            // in c code: output_int
            let bytes = bwlzh_buf_len.to_ne_bytes();
            output[outdata..outdata + 4].copy_from_slice(&bytes);
            output[outdata..outdata + bwlzh_buf_len].copy_from_slice(&bwlzh_buf);
            outdata += bwlzh_buf_len;
        }

        // in c code: output_int
        let bytes = xtc3_context.nrle.to_ne_bytes();
        output[outdata..outdata + 4].copy_from_slice(&bytes);
        outdata += 4;

        if xtc3_context.nrle > 0 {
            let mut bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.nrle)];
            let bwlzh_buf_len = if *speed >= 5 {
                bwlzh_compress(&xtc3_context.rle, xtc3_context.nrle, &mut bwlzh_buf)
            } else {
                bwlzh_compress_no_lz77(&xtc3_context.rle, xtc3_context.nrle, &mut bwlzh_buf)
            };
            // in c code: output_int
            let bytes = bwlzh_buf_len.to_ne_bytes();
            output[outdata..outdata + 4].copy_from_slice(&bytes);
            output[outdata..outdata + bwlzh_buf_len].copy_from_slice(&bwlzh_buf);
            outdata += bwlzh_buf_len;
        }

        // in c code: output_int
        let bytes = xtc3_context.nlargedir.to_ne_bytes();
        output[outdata..outdata + 4].copy_from_slice(&bytes);
        outdata += 4;

        let mut bwlzh_buf;
        let mut bwlzh_buf_len;
        if xtc3_context.nlargedir > 0 {
            if *speed <= 2
                || (*speed <= 5
                    && (!heuristic_bwlzh(&xtc3_context.large_direct, xtc3_context.nlargedir)))
            {
                bwlzh_buf = vec![];
                bwlzh_buf_len = usize::try_from(i32::MAX).expect("usize from i32");
            } else {
                bwlzh_buf = vec![0; bwlzh_get_buflen(xtc3_context.nlargedir)];
                bwlzh_buf_len = if *speed >= 5 {
                    bwlzh_compress(
                        &xtc3_context.large_direct,
                        xtc3_context.nlargedir,
                        &mut bwlzh_buf,
                    )
                } else {
                    bwlzh_compress_no_lz77(
                        &xtc3_context.large_direct,
                        xtc3_context.nlargedir,
                        &mut bwlzh_buf,
                    )
                };
            }
            // If this can be written using base compression we should do that
            // let (base_buf, base_buf_len) =
            //     base_compress(xtc3_context.large_direct, xtc3_context.nlargedir);
        }
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
