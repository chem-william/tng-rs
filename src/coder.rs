use crate::{
    bwlzh::{bwlzh_compress, bwlzh_compress_no_lz77, bwlzh_get_buflen},
    compress::{
        TNG_COMPRESS_ALGO_BWLZH1, TNG_COMPRESS_ALGO_BWLZH2, TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA,
        TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE, TNG_COMPRESS_ALGO_POS_XTC2,
        TNG_COMPRESS_ALGO_POS_XTC3, TNG_COMPRESS_ALGO_STOPBIT, TNG_COMPRESS_ALGO_TRIPLET,
    },
    xtc2::ptngc_pack_array_xtc2,
    xtc3::{positive_int, ptngc_pack_array_xtc3},
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
            let extract = !(0xffffffffu32 << coding_parameter);
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
        n_atoms: usize,
    ) -> bool {
        let mut new_parameter = -1;
        let mut best_length = 0;
        for bits in 1..20 {
            let result = self.pack_array(
                input,
                length,
                TNG_COMPRESS_ALGO_TRIPLET,
                bits,
                n_atoms,
                &mut 0,
            );
            if let Some((_, packed)) = result
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
        n_atoms: usize,
    ) -> bool {
        let mut new_parameter = -1;
        let mut best_length = 0;
        for bits in 1..20 {
            let result = self.pack_array(
                input,
                length,
                TNG_COMPRESS_ALGO_STOPBIT,
                bits,
                n_atoms,
                &mut 0,
            );
            if let Some((_, packed)) = result
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
}
