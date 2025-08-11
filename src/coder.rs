use crate::{
    bwlzh::{bwlzh_compress, bwlzh_compress_no_lz77, bwlzh_get_buflen},
    compress::{TNG_COMPRESS_ALGO_BWLZH1, TNG_COMPRESS_ALGO_BWLZH2, TNG_COMPRESS_ALGO_POS_XTC3},
    fix_point::FixT,
    xtc3::ptngc_pack_array_xtc3,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Coder {
    pub(crate) pack_temporary: u32,
    pub(crate) pack_temporary_bits: i32,
    pub(crate) stat_overflow: i32,
    pub(crate) stat_numval: i32,
}

impl Coder {
    // c version
    pub(crate) fn out8bits_vec(&mut self, out: &mut Vec<u8>) {
        while self.pack_temporary_bits >= 8 {
            self.pack_temporary_bits -= 8;
            out.push(((self.pack_temporary >> self.pack_temporary_bits) & 0xFF) as u8);
            self.pack_temporary &= !(0xFFu32 << self.pack_temporary_bits);
        }
    }

    // c version: Ptngc_pack_array
    pub(crate) fn pack_array(
        &mut self,
        input: &[i32],
        length: &mut usize,
        coding: i32,
        coding_parameter: i32,
        n_atoms: usize,
        speed: usize,
    ) -> (Vec<u8>, usize) {
        unimplemented!();
        // match coding {
        //     TNG_COMPRESS_ALGO_BWLZH1 | TNG_COMPRESS_ALGO_BWLZH2 => {
        //         let mut output = vec![0; 4 + bwlzh_get_buflen(*length)];
        //         let i = *length;
        //         let j = *length;
        //         let k = *length;
        //         let n = *length;
        //         let n_frames = n / n_atoms / 3;
        //         let mut cnt = 0;
        //         let mut pval: Vec<u32> = vec![0; n];

        //         let mut most_negative = FixT::MAX31BIT as i32;
        //         for i in 0..n {
        //             if input[i] < most_negative {
        //                 most_negative = input[i];
        //             }
        //         }
        //         most_negative = -most_negative;
        //         let bytes = (most_negative as u32).to_le_bytes();
        //         output[0..4].copy_from_slice(&bytes);

        //         for i in 0..n_atoms {
        //             for j in 0..3 {
        //                 for k in 0..n_frames {
        //                     let item = input[k * 3 * n_atoms + i * 3 + j];
        //                     pval[cnt] = u32::try_from(item + most_negative).expect("u32 from i32");
        //                     cnt += 1;
        //                 }
        //             }
        //         }

        //         if speed >= 5 {
        //             let output_len = bwlzh_compress(&pval, n, &mut output[4..]);
        //         } else {
        //             let output_len = bwlzh_compress_no_lz77(&pval, n, &mut output[4..]);
        //         }
        //         *length += 4;

        //         return output;
        //     }
        //     TNG_COMPRESS_ALGO_POS_XTC3 => {
        //         return ptngc_pack_array_xtc3(&mut input, &mut length, n_atoms, &mut speed);
        //     }
        //     TNG_COMPRESS_ALGO_POS_XTC2 => {
        //         return ptngc_pack_array_xtc2();
        //     }
        //     _ => {}
        // }
    }
}
