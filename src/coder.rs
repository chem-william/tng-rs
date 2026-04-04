use log::debug;

use crate::{
    TngError,
    bwlzh::{bwlzh_compress, bwlzh_compress_no_lz77, bwlzh_decompress, bwlzh_get_buflen},
    compress::{
        TNG_COMPRESS_ALGO_BWLZH1, TNG_COMPRESS_ALGO_BWLZH2, TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA,
        TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE, TNG_COMPRESS_ALGO_POS_XTC2,
        TNG_COMPRESS_ALGO_POS_XTC3, TNG_COMPRESS_ALGO_STOPBIT, TNG_COMPRESS_ALGO_TRIPLET,
        TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER,
    },
    xtc2::{
        INSTR_BASE_RUNLENGTH, INSTR_DEFAULT, INSTR_FLIP, INSTR_LARGE_BASE_CHANGE, INSTR_LARGE_RLE,
        INSTR_ONLY_LARGE, INSTR_ONLY_SMALL, INSTRNAMES, MAGIC_BITS, compute_magic_bits,
        ptngc_pack_array_xtc2, read_instruction, readbits, readmanybits, trajcoder_base_decompress,
    },
    xtc3::{
        positive_int, ptngc_pack_array_xtc3, ptngc_unpack_array_xtc3, swap_ints, unpositive_int,
    },
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Coder {
    pub(crate) pack_temporary: u32,
    pub(crate) pack_temporary_bits: i32,
    pub(crate) stat_overflow: i32,
    pub(crate) stat_numval: i32,
}

impl Coder {
    pub(crate) fn ptngc_pack_flush(&mut self, output: &mut Vec<u8>) {
        // Zero-fill just enough.
        if self.pack_temporary_bits > 0 {
            self.ptngc_write_pattern(0, 8 - self.pack_temporary_bits, output);
        }
    }
    // c version
    pub(crate) fn out8bits(&mut self, out: &mut Vec<u8>) {
        while self.pack_temporary_bits >= 8 {
            self.pack_temporary_bits -= 8;
            out.push(((self.pack_temporary >> self.pack_temporary_bits) & 0xFF) as u8);
            self.pack_temporary &= !(0xFFu32 << self.pack_temporary_bits);
        }
    }

    pub(crate) fn ptngc_writebits(&mut self, value: u32, nbits: u32, output: &mut Vec<u8>) {
        // Make room for the bits
        self.pack_temporary <<= nbits;
        self.pack_temporary_bits += i32::try_from(nbits).expect("i32 from u32");
        self.pack_temporary |= value;
        self.out8bits(output);
    }

    // c code: ptngc_writemanybits
    // Write "arbitrary" number of bits
    pub(crate) fn ptngc_write_many_bits(
        &mut self,
        value: &[u8],
        mut nbits: u32,
        output: &mut Vec<u8>,
    ) {
        let mut vptr = 0;
        while nbits >= 24 {
            let v = ((u32::from(value[vptr])) << 16)
                | ((u32::from(value[vptr + 1])) << 8)
                | (u32::from(value[vptr + 2]));
            self.ptngc_writebits(v, 24, output);
            vptr += 3;
            nbits -= 24;
        }

        while nbits >= 8 {
            self.ptngc_writebits(u32::from(value[vptr]), 8, output);
            vptr += 1;
            nbits -= 8;
        }

        if nbits > 0 {
            self.ptngc_writebits(u32::from(value[vptr]), nbits, output);
        }
    }

    // c version: Ptngc_pack_array
    pub(crate) fn pack_array(
        &mut self,
        input: &mut [i32],
        length: &mut usize,
        coding: i32,
        coding_parameter: i32,
        n_atoms: usize,
        speed: &mut usize,
    ) -> Option<(Vec<u8>, usize)> {
        match coding {
            TNG_COMPRESS_ALGO_BWLZH1 | TNG_COMPRESS_ALGO_BWLZH2 => {
                let mut output = vec![0; 4 + bwlzh_get_buflen(*length)];
                let n = *length;
                let n_frames = n / n_atoms / 3;
                let mut cnt = 0;
                let mut pval: Vec<u32> = vec![0; n];

                // let mut most_negative = FixT::MAX31BIT as i32;
                let mut most_negative = input
                    .iter()
                    .take(n)
                    .copied()
                    .min()
                    .expect("a negative number");
                most_negative = most_negative.wrapping_neg();
                let bytes = (most_negative as u32).to_le_bytes();
                output[0..4].copy_from_slice(&bytes);

                for i in 0..n_atoms {
                    for j in 0..3 {
                        for k in 0..n_frames {
                            let item = input[k * 3 * n_atoms + i * 3 + j];
                            pval[cnt] = item.wrapping_add(most_negative) as u32;
                            cnt += 1;
                        }
                    }
                }

                let mut output_len = if *speed >= 5 {
                    bwlzh_compress(&pval, n, &mut output[4..])
                } else {
                    bwlzh_compress_no_lz77(&pval, n, &mut output[4..])
                };
                output_len += 4;

                Some((output, output_len))
            }
            TNG_COMPRESS_ALGO_POS_XTC3 => {
                Some(ptngc_pack_array_xtc3(input, length, n_atoms, speed))
            }
            TNG_COMPRESS_ALGO_POS_XTC2 => Some(ptngc_pack_array_xtc2(self, input, length)),
            _ => {
                // Allocate enough memory for output
                let mut output = Vec::with_capacity(8 * *length);

                self.stat_numval = 0;
                self.stat_overflow = 0;
                match coding {
                    TNG_COMPRESS_ALGO_TRIPLET
                    | TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA
                    | TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE => {
                        // Pack triplets
                        let ntriplets = *length / 3;
                        // Determine max base and maxbits
                        let mut max_base = 1 << coding_parameter;
                        let mut maxbits = coding_parameter;
                        let mut intmax = 0;
                        for item in input.iter().take(*length) {
                            let s = positive_int(*item);
                            if s > intmax {
                                intmax = s;
                            }
                        }
                        // Store intmax
                        self.pack_temporary_bits = 32;
                        self.pack_temporary = intmax;
                        self.out8bits(&mut output);
                        while intmax >= max_base {
                            max_base *= 2;
                            maxbits += 1;
                        }
                        for i in 0..ntriplets {
                            let mut s = [0; 3];
                            for (j, s_j) in s.iter_mut().enumerate() {
                                let item = input[i * 3 + j];
                                // Find this symbol in table
                                *s_j = positive_int(item);
                            }
                            if self.pack_triplet(
                                &s,
                                &mut output,
                                coding_parameter,
                                max_base,
                                maxbits,
                            ) {
                                return None;
                            }
                        }
                    }
                    _ => {
                        for &item in input.iter().take(*length) {
                            if self.pack_stopbits_item(item, &mut output, coding_parameter) {
                                return None;
                            }
                        }
                    }
                }
                self.ptngc_pack_flush(&mut output);
                let output_length = output.len();

                Some((output, output_length))
            }
        }
    }

    fn ptngc_write_pattern(&mut self, pattern: i32, nbits: i32, output: &mut Vec<u8>) {
        let mut tmp_nbits = nbits;
        let mut mask1 = 1;
        let mut mask2 = 1 << (tmp_nbits - 1);
        self.pack_temporary <<= tmp_nbits; // Make room for new data
        self.pack_temporary_bits += tmp_nbits;
        while tmp_nbits > 0 {
            if pattern & mask1 > 0 {
                self.pack_temporary |= mask2;
            }
            tmp_nbits -= 1;
            mask1 <<= 1;
            mask2 >>= 1;
        }
        self.out8bits(output);
    }

    pub(crate) fn unpack_array_bwlzh<'a>(
        &self,
        packed: &'a [u8],
        output: &'a mut [i32],
        length: i32,
        n_atoms: usize,
    ) {
        let n = length;
        let nframes = n as usize / n_atoms / 3;
        let mut pval = vec![0u32; n as usize];
        let mut cnt = 0;
        let most_negative = (u32::from(packed[0])
            | (u32::from(packed[1]) << 8)
            | (u32::from(packed[2]) << 16)
            | (u32::from(packed[3]) << 24)) as i32;

        bwlzh_decompress(&packed[4..], length, &mut pval);

        for i in 0..n_atoms {
            for j in 0..3 {
                for k in 0..nframes {
                    let s = pval[cnt];
                    cnt += 1;
                    output[k * 3 * n_atoms + i * 3 + j] = s as i32 - most_negative;
                }
            }
        }
    }

    // C API: Ptngc_unpack_array coder.c line 592
    pub(crate) fn unpack_array(
        &self,
        packed: &[u8],
        output: &mut [i32],
        length: i32,
        coding: i32,
        coding_parameter: i32,
        n_atoms: usize,
    ) -> Result<(), TngError> {
        match coding {
            TNG_COMPRESS_ALGO_STOPBIT | TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER => {
                self.unpack_array_stop_bits(packed, output, length, coding_parameter);
                Ok(())
            }

            TNG_COMPRESS_ALGO_TRIPLET
            | TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA
            | TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE => {
                self.unpack_array_triplet(packed, output, length, coding_parameter);
                Ok(())
            }
            TNG_COMPRESS_ALGO_POS_XTC2 => {
                self.unpack_array_xtc2(packed, output, length);
                Ok(())
            }
            TNG_COMPRESS_ALGO_BWLZH1 | TNG_COMPRESS_ALGO_BWLZH2 => {
                self.unpack_array_bwlzh(packed, output, length, n_atoms);
                Ok(())
            }
            TNG_COMPRESS_ALGO_POS_XTC3 => {
                ptngc_unpack_array_xtc3(packed, output, length, n_atoms);
                Ok(())
            }
            _ => Err(TngError::Constraint(format!(
                "encountered unknown unpack algorithm. Got {coding}"
            ))),
        }
    }

    fn pack_triplet(
        &mut self,
        s: &[u32],
        output: &mut Vec<u8>,
        coding_parameter: i32,
        max_base: u32,
        maxbits: i32,
    ) -> bool {
        // Determine base for this triplet
        let min_base = 1 << coding_parameter;
        let mut this_base = min_base;
        let mut jbase: u32 = 0;
        let mut bits_per_value;
        for &s_i in s.iter().take(3) {
            while s_i >= this_base {
                this_base *= 2;
                jbase += 1;
            }
        }
        bits_per_value = u32::try_from(coding_parameter).expect("u32 from i32") + jbase;
        if jbase >= 3 {
            if this_base > max_base {
                return true;
            }
            bits_per_value = u32::try_from(maxbits).expect("u32 from i32");
            jbase = 3;
        }

        // 2 bits selects the base
        self.pack_temporary <<= 2;
        self.pack_temporary_bits += 2;
        self.pack_temporary |= jbase;
        self.out8bits(output);
        for &s_i in s.iter().take(3) {
            self.ptngc_write32bits(s_i, bits_per_value, output);
        }

        false
    }

    /// Write up to 32 bits
    fn ptngc_write32bits(&mut self, value: u32, mut nbits: u32, output: &mut Vec<u8>) {
        let mut mask = if nbits >= 8 {
            0xFF << (nbits - 8)
        } else {
            0xFF >> (8 - nbits)
        };

        while nbits > 8 {
            // Make room for the bits
            nbits -= 8;
            self.pack_temporary <<= 8;
            self.pack_temporary_bits += 8;
            self.pack_temporary |= (value & mask) >> nbits;
            self.out8bits(output);
            mask >>= 8;
        }

        if nbits > 0 {
            self.ptngc_writebits(value & mask, nbits, output);
        }
    }

    fn pack_stopbits_item(
        &mut self,
        item: i32,
        output: &mut Vec<u8>,
        coding_parameter: i32,
    ) -> bool {
        // Find this symbol in table
        let s = positive_int(item);

        self.write_stop_bit_code(s, coding_parameter, output)
    }

    fn write_stop_bit_code(
        &mut self,
        mut s: u32,
        mut coding_parameter: i32,
        output: &mut Vec<u8>,
    ) -> bool {
        loop {
            let extract = !(0xffff_ffff_u32 << coding_parameter);
            let mut this = (s & extract) << 1;
            s >>= coding_parameter;
            if s > 0 {
                this |= 1;
                self.stat_overflow += 1;
            }
            self.pack_temporary <<= coding_parameter + 1;
            self.pack_temporary_bits += coding_parameter + 1;
            self.pack_temporary |= this;
            self.out8bits(output);
            if s > 0 {
                coding_parameter >>= 1;
                if coding_parameter < 1 {
                    coding_parameter = 1;
                }
            }
            if s == 0 {
                break;
            }
        }
        self.stat_numval += 1;
        false
    }

    pub(crate) fn determine_best_coding_triple(
        &mut self,
        input: &mut [i32],
        length: &mut usize,
        coding_parameter: &mut i32,
        _n_atoms: usize,
    ) -> bool {
        let n = *length;
        let ntriplets = n / 3;

        // Precompute positive_int values and per-triplet max
        let positive: Vec<u32> = input[..n].iter().map(|&v| positive_int(v)).collect();
        let intmax = positive.iter().copied().max().unwrap_or(0);

        // Precompute the max of each triplet
        let triplet_max: Vec<u32> = (0..ntriplets)
            .map(|i| {
                let a = positive[i * 3];
                let b = positive[i * 3 + 1];
                let c = positive[i * 3 + 2];
                a.max(b).max(c)
            })
            .collect();

        let mut new_parameter = -1;
        let mut best_length = 0;

        for bits in 1..20i32 {
            let packed = estimate_triplet_size(intmax, &triplet_max, bits);
            if let Some(packed) = packed
                && packed > 0
                && (new_parameter == -1 || packed < best_length)
            {
                new_parameter = bits;
                best_length = packed;
            }
        }

        if new_parameter == -1 {
            true
        } else {
            *coding_parameter = new_parameter;
            *length = best_length;
            false
        }
    }

    pub(crate) fn determine_best_coding_stop_bits(
        &mut self,
        input: &mut [i32],
        length: &mut usize,
        coding_parameter: &mut i32,
        _n_atoms: usize,
    ) -> bool {
        let n = *length;
        // Precompute bit-widths of positive_int values (1 byte each, cache-friendly)
        let bitwidths: Vec<u8> = input[..n]
            .iter()
            .map(|&v| (32 - positive_int(v).leading_zeros()) as u8)
            .collect();

        let mut new_parameter = -1;
        let mut best_length = 0;
        for bits in 1..20 {
            let lut = build_stopbit_lut(bits);
            let mut total_bits: usize = 0;
            for &bw in &bitwidths {
                total_bits += lut[bw as usize] as usize;
            }
            let packed = (total_bits + 7) / 8;
            if packed > 0 && (new_parameter == -1 || packed < best_length) {
                new_parameter = bits;
                best_length = packed;
            }
        }

        if new_parameter == -1 {
            true
        } else {
            *coding_parameter = new_parameter;
            *length = best_length;
            false
        }
    }
}

/// Compute stop-bit bits needed for a value with `bw` significant bits at a given coding_parameter.
fn stopbit_bits_for_bitwidth(mut bw: u32, mut cp: i32) -> u32 {
    let mut total = 0u32;
    loop {
        total += (cp + 1) as u32; // cp data bits + 1 stop bit
        if bw as i32 <= cp {
            break;
        }
        bw -= cp as u32;
        cp >>= 1;
        if cp < 1 {
            cp = 1;
        }
    }
    total
}

/// Build lookup table: bits_cost[bw] = total stop-bit bits for a value with `bw` significant bits.
fn build_stopbit_lut(coding_parameter: i32) -> [u32; 33] {
    let mut lut = [0u32; 33];
    for bw in 0..33u32 {
        lut[bw as usize] = stopbit_bits_for_bitwidth(bw, coding_parameter);
    }
    lut
}

/// Estimate the output size in bytes of stop-bit encoding without actually encoding.
fn estimate_stopbit_size(positive_vals: &[u32], coding_parameter: i32) -> usize {
    let lut = build_stopbit_lut(coding_parameter);
    let mut total_bits: usize = 0;
    for &s in positive_vals {
        let bw = 32 - s.leading_zeros();
        total_bits += lut[bw as usize] as usize;
    }
    // Round up to bytes
    (total_bits + 7) / 8
}

/// Estimate the output size in bytes of triplet encoding without actually encoding.
/// Returns None if encoding would fail (value exceeds max_base).
fn estimate_triplet_size(intmax: u32, triplet_max: &[u32], coding_parameter: i32) -> Option<usize> {
    let cp = coding_parameter as u32;
    let mut max_base: u32 = 1u32.checked_shl(cp)?;
    let mut maxbits = cp;
    {
        let mut tmp = intmax;
        while tmp >= max_base {
            max_base = max_base.checked_mul(2)?;
            maxbits += 1;
        }
    }

    // Build LUT: for bit-width bw of triplet max, what are the total bits per triplet?
    // bw 0..=cp → jbase=0, bits = 2 + 3*cp
    // bw cp+1   → jbase=1, bits = 2 + 3*(cp+1)
    // bw cp+2   → jbase=2, bits = 2 + 3*(cp+2)
    // bw cp+3..=maxbits → jbase=3, bits = 2 + 3*maxbits
    // bw > maxbits → fail (use sentinel 0)
    let mut lut = [0u32; 33];
    for bw in 0..33u32 {
        if bw <= cp {
            lut[bw as usize] = 2 + 3 * cp;
        } else if bw == cp + 1 {
            lut[bw as usize] = 2 + 3 * (cp + 1);
        } else if bw == cp + 2 {
            lut[bw as usize] = 2 + 3 * (cp + 2);
        } else if bw <= maxbits {
            lut[bw as usize] = 2 + 3 * maxbits;
        }
        // else: lut[bw] stays 0 (sentinel for failure)
    }

    // 32 bits for intmax header
    let mut total_bits: usize = 32;

    for &tmax in triplet_max {
        let bw = 32 - tmax.leading_zeros();
        let bits = lut[bw as usize];
        if bits == 0 && tmax > 0 {
            return None;
        }
        total_bits += bits as usize;
    }

    // Round up to bytes (matching ptngc_pack_flush behavior)
    Some((total_bits + 7) / 8)
}

impl Coder {
    pub(crate) fn unpack_array_stop_bits<'a>(
        &self,
        packed: &'a [u8],
        output: &'a mut [i32],
        length: i32,
        coding_parameter: i32,
    ) {
        let mut extract_mask = 0x80;
        let ptr = packed;
        let mut ptr_count = 0;

        for i in 0..length {
            let mut pattern = 0;
            let mut numbits = coding_parameter;
            let mut bit;
            let mut insert_mask = (1u32).checked_shl((numbits - 1) as u32).unwrap_or(0);
            // let mut insert_mask = 1u32 << (numbits - 1);
            let mut inserted_bits = numbits;

            loop {
                for _ in 0..numbits {
                    bit = ptr[ptr_count] & extract_mask;
                    if bit > 0 {
                        pattern |= insert_mask;
                    }
                    insert_mask >>= 1;
                    extract_mask >>= 1;
                    if extract_mask == 0 {
                        extract_mask = 0x80;
                        ptr_count += 1;
                    }
                }
                // Check stop bit
                bit = ptr[ptr_count] & extract_mask;
                extract_mask >>= 1;
                if extract_mask == 0 {
                    extract_mask = 0x80;
                    ptr_count += 1;
                }
                if bit > 0 {
                    numbits >>= 1;
                    if numbits < 1 {
                        numbits = 1;
                    }
                    inserted_bits += numbits;
                    insert_mask = 1u32 << (inserted_bits - 1);
                }
                if bit == 0 {
                    break;
                }
            }
            let mut s = (pattern).div_ceil(2) as i32;
            if pattern % 2 == 0 {
                s = -s;
            }
            output[i as usize] = s;
        }
    }

    pub(crate) fn unpack_array_triplet<'a>(
        &self,
        packed: &'a [u8],
        output: &'a mut [i32],
        length: i32,
        coding_parameter: i32,
    ) {
        let mut extract_mask = 0x80;
        let mut max_base = 1u32 << coding_parameter;
        let mut maxbits = coding_parameter;
        let ptr = packed;
        let intmax = (u32::from(ptr[0]) << 24)
            | (u32::from(ptr[1]) << 16)
            | (u32::from(ptr[2]) << 8)
            | u32::from(ptr[3]);
        let mut ptr_count = 4;

        while intmax >= max_base {
            max_base *= 2;
            maxbits += 1;
        }

        for i in 0..(length / 3) {
            // Find base
            let mut jbase = 0;
            let mut bit;
            for _ in 0..2 {
                bit = ptr[ptr_count] & extract_mask;
                jbase <<= 1;
                if bit != 0 {
                    jbase |= 1u32;
                }
                extract_mask >>= 1;
                if extract_mask == 0 {
                    extract_mask = 0x80;
                    ptr_count += 1;
                }
            }
            let numbits = if jbase == 3 {
                maxbits
            } else {
                coding_parameter + jbase as i32
            };
            for j in 0..3 {
                let mut pattern = 0;
                for _ in 0..numbits {
                    bit = ptr[ptr_count] & extract_mask;
                    pattern <<= 1;
                    if bit != 0 {
                        pattern |= 1u32;
                    }
                    extract_mask >>= 1;
                    if extract_mask == 0 {
                        extract_mask = 0x80;
                        ptr_count += 1;
                    }
                }
                let mut s = pattern.div_ceil(2) as i32;
                if pattern % 2 == 0 {
                    s = -s;
                }
                output[(i * 3 + j) as usize] = s;
            }
        }
    }

    // C API: Ptngc_unpack_array_xtc2
    pub(crate) fn unpack_array_xtc2<'a>(
        &self,
        packed: &'a [u8],
        output: &'a mut [i32],
        length: i32,
    ) {
        let mut output_counter = 0;
        let mut ptr = packed;
        let mut bitptr = 0;
        let mut ntriplets_left = length / 3;
        let mut swapatoms = false;
        let mut runlength: i32 = 0;
        let mut compress_buffer = [0; 18 * 4]; // Holds compressed result for 3 large ints or up to 18 small ints
        let mut encode_ints = [0; 21]; // Up to 3 large + 18 small ints can be encoded at once

        // Read min integers
        let minint = [
            unpositive_int(readbits(&mut ptr, &mut bitptr, 32) as i32),
            unpositive_int(readbits(&mut ptr, &mut bitptr, 32) as i32),
            unpositive_int(readbits(&mut ptr, &mut bitptr, 32) as i32),
        ];
        // Read large indices
        let large_index = [
            readbits(&mut ptr, &mut bitptr, 8),
            readbits(&mut ptr, &mut bitptr, 8),
            readbits(&mut ptr, &mut bitptr, 8),
        ];
        // Read small index
        let mut small_index = readbits(&mut ptr, &mut bitptr, 8) as i32;

        let large_nbits = compute_magic_bits(large_index);

        debug!(
            "Minimum integers: {} {} {}",
            minint[0], minint[1], minint[2]
        );
        debug!(
            "Large indices: {} {} {}",
            large_index[0], large_index[1], large_index[2]
        );
        debug!("Small index: {small_index}");
        debug!("large_nbits={large_nbits}");

        // Initial prevcoord is the minimum integers
        let mut prevcoord = minint;

        while ntriplets_left != 0 {
            let instr = read_instruction(&mut ptr, &mut bitptr);
            debug!("Decoded instruction {}", INSTRNAMES[instr as usize]);

            if instr == INSTR_DEFAULT || instr == INSTR_ONLY_LARGE || instr == INSTR_ONLY_SMALL {
                let mut large_ints = [0; 3];
                if instr != INSTR_ONLY_SMALL {
                    // Clear the compress buffer
                    compress_buffer.fill(0);
                    // Get the large value
                    readmanybits(
                        &mut ptr,
                        &mut bitptr,
                        large_nbits as i32,
                        &mut compress_buffer,
                    );
                    trajcoder_base_decompress(&compress_buffer, 3, &large_index, &mut encode_ints);
                    large_ints.copy_from_slice(&encode_ints[..3]);
                    debug!("large ints: {large_ints:?}");
                }

                if instr != INSTR_ONLY_LARGE {
                    // The same base is used for the small changes
                    let small_idx = [small_index as u32; 3];

                    // Clear the compress buffer
                    compress_buffer.fill(0);

                    // Get the small values — encoder always sends INSTR_BASE_RUNLENGTH
                    // before any INSTR_DEFAULT/INSTR_ONLY_SMALL, so runlength >= 1 here
                    readmanybits(
                        &mut ptr,
                        &mut bitptr,
                        MAGIC_BITS[small_index as usize][(runlength - 1) as usize] as i32,
                        &mut compress_buffer,
                    );
                    trajcoder_base_decompress(
                        &compress_buffer,
                        3 * runlength,
                        &small_idx,
                        &mut encode_ints,
                    );
                }

                if instr == INSTR_DEFAULT {
                    // Check for swapped atoms
                    if swapatoms {
                        // Unswap the atoms
                        for i in 0..3 {
                            let mut out = [0; 3];
                            let inp = [
                                large_ints[i],
                                unpositive_int(encode_ints[i]),
                                unpositive_int(encode_ints[3 + i]),
                            ];
                            swap_ints(&inp, &mut out);
                            large_ints[i] = out[0];
                            encode_ints[i] = positive_int(out[1]) as i32;
                            encode_ints[3 + i] = positive_int(out[2]) as i32;
                        }
                    }
                }
                // Output result
                if instr != INSTR_ONLY_SMALL {
                    // Output large values
                    output[output_counter] = large_ints[0] + minint[0];
                    output_counter += 1;
                    output[output_counter] = large_ints[1] + minint[1];
                    output_counter += 1;
                    output[output_counter] = large_ints[2] + minint[2];
                    output_counter += 1;

                    prevcoord = large_ints;
                    debug!("Prevcoord after unpacking of large: {prevcoord:?}");
                    debug!(
                        "VALUE: {} {} {} {}",
                        length / 3 - ntriplets_left,
                        prevcoord[0] + minint[0],
                        prevcoord[1] + minint[1],
                        prevcoord[2] + minint[2]
                    );
                    ntriplets_left -= 1;
                }
                if instr != INSTR_ONLY_LARGE {
                    // Output small values
                    debug!("Prevcoord before unpacking of small: {prevcoord:?}");

                    for i in 0..runlength as usize {
                        let v = [
                            unpositive_int(encode_ints[i * 3]),
                            unpositive_int(encode_ints[i * 3 + 1]),
                            unpositive_int(encode_ints[i * 3 + 2]),
                        ];
                        prevcoord[0] += v[0];
                        prevcoord[1] += v[1];
                        prevcoord[2] += v[2];

                        debug!("Prevcoord after unpacking of small: {prevcoord:?}");
                        debug!("Unpacked small values: {v:?} {prevcoord:?}");
                        debug!(
                            "VALUE: {} {} {} {}",
                            length / 3 - (ntriplets_left - i as i32),
                            prevcoord[0] + minint[0],
                            prevcoord[1] + minint[1],
                            prevcoord[2] + minint[2]
                        );

                        output[output_counter] = prevcoord[0] + minint[0];
                        output_counter += 1;
                        output[output_counter] = prevcoord[1] + minint[1];
                        output_counter += 1;
                        output[output_counter] = prevcoord[2] + minint[2];
                        output_counter += 1;
                    }
                    ntriplets_left -= runlength;
                }
            } else if instr == INSTR_LARGE_RLE {
                let mut large_ints = [0; 3];
                // How many large atoms in this sequence?
                let n = (readbits(&mut ptr, &mut bitptr, 4) + 3) as i32; // 3-18 large atoms
                for _ in 0..n as usize {
                    // Clear the compress buffer
                    compress_buffer.fill(0);
                    // Get the large value
                    readmanybits(
                        &mut ptr,
                        &mut bitptr,
                        large_nbits as i32,
                        &mut compress_buffer,
                    );
                    trajcoder_base_decompress(&compress_buffer, 3, &large_index, &mut encode_ints);
                    large_ints.copy_from_slice(&encode_ints[..3]);
                    output[output_counter] = large_ints[0] + minint[0];
                    output_counter += 1;
                    output[output_counter] = large_ints[1] + minint[1];
                    output_counter += 1;
                    output[output_counter] = large_ints[2] + minint[2];
                    output_counter += 1;
                    prevcoord.copy_from_slice(&large_ints);
                }
                ntriplets_left -= n;
            } else if instr == INSTR_BASE_RUNLENGTH {
                let code = readbits(&mut ptr, &mut bitptr, 4) as i32;
                let change;
                if code == 15 {
                    change = 0;
                    runlength = 6;
                } else {
                    let ichange = code % 3;
                    runlength = code / 3 + 1;
                    change = ichange - 1;
                }
                small_index += change;
            } else if instr == INSTR_FLIP {
                swapatoms = !swapatoms;
            } else if instr == INSTR_LARGE_BASE_CHANGE {
                let ichange = readbits(&mut ptr, &mut bitptr, 2);
                let mut change = (ichange & 0x1_u32) as i32 + 1;
                if (ichange & 0x2_u32) != 0 {
                    change = -change;
                }
                small_index += change;
            } else {
                panic!("BUG! Encoded unknown instruction");
            }
            debug!("Number of triplets left is {ntriplets_left}");
        }
    }
}
