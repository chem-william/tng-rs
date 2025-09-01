use crate::{
    coder::Coder,
    fix_point::{FixT, fixt_pair_to_f64},
};

const MAX_FVAL: f32 = 2147483647.0;

// Compression algorithms (matching the original trajng assignments) The compression
// backends require that some of the algorithms must have the same value
pub(crate) const TNG_COMPRESS_ALGO_STOPBIT: i32 = 1;
pub(crate) const TNG_COMPRESS_ALGO_TRIPLET: i32 = 2;
pub(crate) const TNG_COMPRESS_ALGO_BWLZH1: i32 = 8;
pub(crate) const TNG_COMPRESS_ALGO_BWLZH2: i32 = 9;

pub(crate) const TNG_COMPRESS_ALGO_POS_STOPBIT_INTER: i32 = TNG_COMPRESS_ALGO_STOPBIT;
pub(crate) const TNG_COMPRESS_ALGO_POS_TRIPLET_INTER: i32 = TNG_COMPRESS_ALGO_TRIPLET;
pub(crate) const TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA: i32 = 3;
pub(crate) const TNG_COMPRESS_ALGO_POS_XTC2: i32 = 5;
pub(crate) const TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE: i32 = 7;
pub(crate) const TNG_COMPRESS_ALGO_POS_BWLZH_INTER: i32 = TNG_COMPRESS_ALGO_BWLZH1;
pub(crate) const TNG_COMPRESS_ALGO_POS_BWLZH_INTRA: i32 = TNG_COMPRESS_ALGO_BWLZH2;
pub(crate) const TNG_COMPRESS_ALGO_POS_XTC3: i32 = 10;

pub(crate) const TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE: i32 = TNG_COMPRESS_ALGO_STOPBIT;
pub(crate) const TNG_COMPRESS_ALGO_VEL_TRIPLET_INTER: i32 = TNG_COMPRESS_ALGO_TRIPLET;
pub(crate) const TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE: i32 = 3;
pub(crate) const TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER: i32 = 6;
pub(crate) const TNG_COMPRESS_ALGO_VEL_BWLZH_INTER: i32 = TNG_COMPRESS_ALGO_BWLZH1;
pub(crate) const TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE: i32 = TNG_COMPRESS_ALGO_BWLZH2;
pub(crate) const TNG_COMPRESS_ALGO_MAX: i32 = 11;

// This becomes TNGP for positions (little endian) and TNGV for velocities. In ASCII
const MAGIC_INT_POS: u32 = 0x50474E54;
const MAGIC_INT_VEL: u32 = 0x56474E54;

#[inline]
pub fn precision(hi: FixT, lo: FixT) -> f64 {
    fixt_pair_to_f64(hi, lo)
}

pub(crate) fn quantize_float(
    x: &[f32],
    n_atoms: usize,
    n_frames: usize,
    precision: f32,
) -> Result<Vec<i32>, ()> {
    let total = n_atoms
        .checked_mul(n_frames)
        .and_then(|v| v.checked_mul(3))
        .expect("overflow computing quant length");
    let mut quant: Vec<i32> = Vec::with_capacity(total);

    for iframe in 0..n_frames {
        for i in 0..n_atoms {
            for j in 0..3 {
                quant[iframe * n_atoms * 3 + i * 3 + j] =
                    ((x[iframe * n_atoms * 3 + i * 3 + j] / precision) + 0.5).floor() as i32;
            }
        }
    }

    if verify_input_data_float(x, n_atoms, n_frames, precision).is_ok() {
        Ok(quant)
    } else {
        Err(())
    }
}

fn verify_input_data_float(
    x: &[f32],
    n_atoms: usize,
    n_frames: usize,
    precision: f32,
) -> Result<(), ()> {
    for iframe in 0..n_frames {
        for i in 0..n_atoms {
            for j in 0..3 {
                if (x[iframe * n_atoms * 3 + i * 3 + j] / precision + 0.5).abs() >= MAX_FVAL {
                    return Err(());
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn quant_inter_differences(quant: &[i32], n_atoms: usize, n_frames: usize) -> Vec<i32> {
    let mut quant_inter = vec![0; n_atoms * n_frames * 3];
    // The first frame is used for absolute positions.
    for i in 0..n_atoms {
        for j in 0..3 {
            quant_inter[i * 3 + j] = quant[i * 3 + j];
        }
    }

    // For all other frames, the difference to the previous frame is used.
    for iframe in 1..n_frames {
        for i in 0..n_atoms {
            for j in 0..3 {
                quant_inter[iframe * n_atoms * 3 + i * 3 + j] = quant
                    [iframe * n_atoms * 3 + i * 3 + j]
                    - quant[(iframe - 1) * n_atoms * 3 + i * 3 + j];
            }
        }
    }
    quant_inter
}

pub(crate) fn quant_intra_differences(quant: &[i32], n_atoms: usize, n_frames: usize) -> Vec<i32> {
    let mut quant_intra = vec![0; n_atoms * n_frames * 3];

    for iframe in 0..n_frames {
        // The first atom is used with its absolute position
        for j in 0..3 {
            quant_intra[iframe * n_atoms * 3 + j] = quant[iframe * n_atoms * 3 + j];
        }

        // For all other atoms the intraframe differences are computed
        for i in 1..n_atoms {
            for j in 0..3 {
                quant_intra[iframe * n_atoms * 3 + i * 3 + j] = quant
                    [iframe * n_atoms * 3 + i * 3 + j]
                    - quant[iframe * n_atoms * 3 + (i - 1) * 3 + j];
            }
        }
    }
    quant_intra
}

pub(crate) fn determine_best_pos_initial_coding(
    quant: &mut [i32],
    quant_intra: &mut [i32],
    n_atoms: usize,
    speed: usize,
    prec_hi: FixT,
    prec_lo: FixT,
    initial_coding: i32,
    initial_coding_parameter: i32,
) -> (i32, i32) {
    if initial_coding == -1 {
        let mut best_coding;
        let mut best_coding_parameter;
        let mut best_code_size;
        let mut current_coding;
        let mut current_coding_parameter;
        let mut current_code_size;
        let mut coder;

        // Start with XTC2, it should always work
        current_coding = TNG_COMPRESS_ALGO_POS_XTC2;
        current_coding_parameter = 0;
        current_code_size = compress_quantized_pos(
            quant,
            None,
            Some(quant_intra),
            n_atoms.try_into().expect("usize from u32"),
            1,
            speed,
            current_coding,
            current_coding_parameter,
            0,
            0,
            prec_hi,
            prec_lo,
            &mut None,
        );
        best_coding = current_coding;
        best_coding_parameter = current_coding_parameter;
        best_code_size = current_code_size;

        // Determine best parameter for triplet intra.
        current_coding = TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA;
        coder = Coder::default();
        current_code_size = n_atoms * 3;
        current_coding_parameter = 0;
        if !coder.determine_best_coding_triple(
            quant_intra,
            &mut current_code_size,
            &mut current_coding_parameter,
            n_atoms,
        ) && current_code_size < best_code_size
        {
            best_coding = current_coding;
            best_coding_parameter = current_coding_parameter;
            best_code_size = current_code_size;
        }

        // Determine best parameter for triplet one-to-one
        current_coding = TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE;
        coder = Coder::default();
        current_code_size = n_atoms * 3;
        current_coding_parameter = 0;
        if !coder.determine_best_coding_triple(
            quant,
            &mut current_code_size,
            &mut current_coding_parameter,
            n_atoms,
        ) && current_code_size < best_code_size
        {
            best_coding = current_coding;
            best_coding_parameter = current_coding_parameter;
            best_code_size = current_code_size;
        }

        if speed >= 2 {
            current_coding = TNG_COMPRESS_ALGO_POS_XTC3;
            current_coding_parameter = 0;
            current_code_size = compress_quantized_pos(
                quant,
                None,
                Some(quant_intra),
                n_atoms.try_into().expect("usize from u32"),
                1,
                speed,
                current_coding,
                current_coding_parameter,
                0,
                0,
                prec_hi,
                prec_lo,
                &mut None,
            );
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
                best_code_size = current_code_size;
            }
        }

        // Test BWLZH intra
        if speed >= 6 {
            current_coding = TNG_COMPRESS_ALGO_POS_BWLZH_INTRA;
            current_coding_parameter = 0;
            current_code_size = compress_quantized_pos(
                quant,
                None,
                Some(quant_intra),
                n_atoms.try_into().expect("usize from u32"),
                1,
                speed,
                current_coding,
                current_coding_parameter,
                0,
                0,
                prec_hi,
                prec_lo,
                &mut None,
            );
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
            }
        }
        (best_coding, best_coding_parameter)
    } else if initial_coding_parameter == -1 {
        match initial_coding {
            TNG_COMPRESS_ALGO_POS_XTC2
            | TNG_COMPRESS_ALGO_POS_XTC3
            | TNG_COMPRESS_ALGO_POS_BWLZH_INTRA => (initial_coding, 0),
            TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA => {
                let mut coder = Coder::default();
                let mut current_code_size = n_atoms * 3;
                let mut resulting_coding_parameter = initial_coding_parameter;
                coder.determine_best_coding_triple(
                    quant_intra,
                    &mut current_code_size,
                    &mut resulting_coding_parameter,
                    n_atoms,
                );
                (initial_coding, resulting_coding_parameter)
            }
            _ => {
                let mut coder = Coder::default();
                let mut current_code_size = n_atoms * 3;
                let mut resulting_coding_parameter = initial_coding_parameter;
                coder.determine_best_coding_triple(
                    quant,
                    &mut current_code_size,
                    &mut resulting_coding_parameter,
                    n_atoms,
                );
                (initial_coding, resulting_coding_parameter)
            }
        }
    } else {
        unreachable!("initial_coding != -1, but initial_coding_parameter != -1")
    }
}

/// Buffer `num` 8 bit bytes into buffer location `buf`
///
/// # Panic
/// Panics if `buf.len() < num`.
fn bufferfix(buf: &mut [u8], v: FixT, num: usize) {
    if buf.len() < num {
        panic!("Buffer too small for requested number of bytes");
    }

    // Store in little endian format
    let mut v: u32 = v.into(); // Convert to unsigned for bit operations
    let mut c: u8; // at least 8 bits
    c = (v & 0xFF) as u8;

    for buf_item in buf.iter_mut().take(num) {
        *buf_item = c;
        v >>= 8;
        c = (v & 0xFF) as u8;
    }
}

/// Perform position compression from the quantized data
fn compress_quantized_pos(
    quant: &mut [i32],
    quant_inter: Option<&mut [i32]>,
    mut quant_intra: Option<&mut [i32]>,
    n_atoms: u32,
    n_frames: u32,
    mut speed: usize,
    initial_coding: i32,
    initial_coding_parameter: i32,
    coding: i32,
    coding_parameter: i32,
    prec_hi: FixT,
    prec_lo: FixT,
    data: &mut Option<&mut [u8]>,
) -> usize {
    let datablock;

    let mut bufloc = 0;
    // Information needed for decompression
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], FixT::from(MAGIC_INT_POS), 4);
    }
    bufloc += 4;

    // Number of atoms
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], FixT::from(n_atoms), 4);
    }
    bufloc += 4;

    // Number of frames
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], FixT::from(n_frames), 4);
    }
    bufloc += 4;

    // Initial coding
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(initial_coding).expect("u32 from i32")),
            4,
        );
    }
    bufloc += 4;

    // Initial coding parameter
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(initial_coding_parameter).expect("u32 from i32")),
            4,
        );
    }
    bufloc += 4;

    // Coding
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(coding).expect("u32 from i32")),
            4,
        );
    }
    bufloc += 4;

    // Coding parameter
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(coding_parameter).expect("u32 from i32")),
            4,
        );
    }
    bufloc += 4;

    // Precision
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], prec_lo, 4);
    }
    bufloc += 4;
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], prec_hi, 4);
    }
    bufloc += 4;

    // The initial frame
    let output_length;
    let length = n_atoms * 3;
    match initial_coding {
        TNG_COMPRESS_ALGO_POS_XTC2
        | TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE
        | TNG_COMPRESS_ALGO_POS_XTC3 => {
            let mut coder = Coder::default();
            (datablock, output_length) = coder
                .pack_array(
                    quant,
                    &mut length.try_into().expect("usize from u32"),
                    initial_coding,
                    initial_coding_parameter,
                    n_atoms.try_into().expect("usize from u32"),
                    &mut speed,
                )
                .expect("packed array");
        }
        TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA | TNG_COMPRESS_ALGO_POS_BWLZH_INTRA => {
            let mut coder = Coder::default();
            (datablock, output_length) = coder
                .pack_array(
                    quant_intra.as_mut().expect("quant_intra to be Some"),
                    &mut length.try_into().expect("usize from u32"),
                    initial_coding,
                    initial_coding_parameter,
                    n_atoms.try_into().expect("usize from u32"),
                    &mut speed,
                )
                .expect("packed array");
        }
        _ => {
            unreachable!()
        }
    }
    // Block length
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(output_length).expect("u32 from usize")),
            4,
        );
    }
    bufloc += 4;
    // The actual data block
    if let Some(mut_data) = data.as_mut() {
        mut_data[bufloc..bufloc + output_length].copy_from_slice(&datablock[..output_length]);
    }
    bufloc += output_length;

    // The remaining frames
    if n_frames > 1 {
        let us_natoms = usize::try_from(n_atoms).expect("usize from u32");
        let mut fallback_len = us_natoms
            .checked_mul(3)
            .and_then(|v| v.checked_mul(usize::try_from(n_frames - 1).expect("usize from u32")))
            .expect("fallback_len overflow");
        let result = match coding {
            // Inter-frame compression
            TNG_COMPRESS_ALGO_POS_STOPBIT_INTER
            | TNG_COMPRESS_ALGO_POS_TRIPLET_INTER
            | TNG_COMPRESS_ALGO_POS_BWLZH_INTER => {
                let mut coder = Coder::default();
                coder.pack_array(
                    &mut quant_inter.expect("quant_inter to be Some")[us_natoms * 3..],
                    &mut fallback_len,
                    coding,
                    coding_parameter,
                    us_natoms,
                    &mut speed,
                )
            }
            // One-to-one compression?
            TNG_COMPRESS_ALGO_POS_XTC2
            | TNG_COMPRESS_ALGO_POS_XTC3
            | TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE => {
                let mut coder = Coder::default();
                coder.pack_array(
                    &mut quant[us_natoms * 3..],
                    &mut fallback_len,
                    coding,
                    coding_parameter,
                    us_natoms,
                    &mut speed,
                )
            }
            // Intra-frame compression?
            TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA | TNG_COMPRESS_ALGO_POS_BWLZH_INTRA => {
                let mut coder = Coder::default();
                coder.pack_array(
                    &mut quant_intra.expect("quant_intra to be Some")[us_natoms * 3..],
                    &mut fallback_len,
                    coding,
                    coding_parameter,
                    us_natoms,
                    &mut speed,
                )
            }
            _ => None,
        };
        // Always have a length and a data slice to write
        let (datablock, output_length) = result.unwrap_or_else(|| (Vec::new(), fallback_len));
        // Block length
        if let Some(mut_data) = data.as_mut() {
            bufferfix(
                &mut mut_data[bufloc..],
                FixT::from(u32::try_from(output_length).expect("u32 from usize")),
                4,
            );
        }
        bufloc += 4;
        if !datablock.is_empty()
            && let Some(mut_data) = data.as_mut()
        {
            mut_data[bufloc..bufloc + output_length].copy_from_slice(&datablock[..output_length]);
        }
        bufloc += output_length;
    }
    bufloc
}

#[cfg(test)]
mod buffer {
    use super::*;

    #[test]
    fn basic_4_byte_value() {
        let mut buf = [0u8; 8];
        let value: FixT = FixT::from(0x12345678);
        bufferfix(&mut buf, value, 4);

        // Expected: little endian format
        assert_eq!(buf[0], 0x78); // LSB first
        assert_eq!(buf[1], 0x56);
        assert_eq!(buf[2], 0x34);
        assert_eq!(buf[3], 0x12); // MSB last
    }

    #[test]
    fn single_byte() {
        let mut buf = [0u8; 8];
        let value: FixT = FixT::from(0xAB);
        bufferfix(&mut buf, value, 1);
        assert_eq!(buf[0], 0xAB);
    }

    #[test]
    fn two_bytes() {
        let mut buf = [0u8; 8];
        let value: FixT = FixT::from(0x1234);
        bufferfix(&mut buf, value, 2);

        assert_eq!(buf[0], 0x34);
        assert_eq!(buf[1], 0x12);
    }

    #[test]
    fn zero_value() {
        let mut buf = [0xFF; 8];
        let value: FixT = FixT::from(0);
        bufferfix(&mut buf, value, 4);
        assert_eq!(buf[0], 0x00);
        assert_eq!(buf[1], 0x00);
        assert_eq!(buf[2], 0x00);
        assert_eq!(buf[3], 0x00);
    }

    #[test]
    fn more_bytes_than_value_size() {
        let mut buf = [0u8; 8];
        let value: FixT = FixT::from(0x1234);
        bufferfix(&mut buf, value, 6);

        assert_eq!(buf[0], 0x34);
        assert_eq!(buf[1], 0x12);
        assert_eq!(buf[2], 0x00); // Higher bytes should be 0
        assert_eq!(buf[3], 0x00);
        assert_eq!(buf[4], 0x00);
        assert_eq!(buf[5], 0x00);
    }

    #[test]
    fn zero_bytes_requested() {
        let mut buf = [0xAA; 8];
        let value: FixT = FixT::from(0x12345678);
        bufferfix(&mut buf, value, 0);

        // Buffer should remain unchanged
        assert_eq!(buf[0], 0xAA);
        assert_eq!(buf[1], 0xAA);
        assert_eq!(buf[2], 0xAA);
        assert_eq!(buf[3], 0xAA);
    }

    #[test]
    fn maximum_value() {
        // Test maximum positive value
        let mut buf = [0u8; 8];
        let value: FixT = FixT::from(FixT::MAX31BIT);
        bufferfix(&mut buf, value, 4);

        assert_eq!(buf[0], 0xFF);
        assert_eq!(buf[1], 0xFF);
        assert_eq!(buf[2], 0xFF);
        assert_eq!(buf[3], 0x7F);
    }
}

#[cfg(test)]
mod compress_quantized_pos_tests {
    use super::*;

    fn u32le(b: &[u8]) -> u32 {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    #[test]
    fn xtc2_then_xtc3() {
        let mut quant = [-1, -2, -3, 10, 20, 30, 5, -10, 42, -7, 0, -999];
        let mut data = [0; 1024];
        let n_atoms = 2;
        let n_frames = 2;
        let speed = 6;
        let initial_coding = TNG_COMPRESS_ALGO_POS_XTC2;
        let initial_coding_parameter = 0;
        let coding = TNG_COMPRESS_ALGO_POS_XTC3;
        let coding_parameter = 0;
        let prec_hi = FixT::from(123);
        let prec_lo = FixT::from(456);
        let nitems = compress_quantized_pos(
            &mut quant,
            None,
            None,
            n_atoms,
            n_frames,
            speed,
            initial_coding,
            initial_coding_parameter,
            coding,
            coding_parameter,
            prec_hi,
            prec_lo,
            &mut Some(&mut data),
        );
        let expected = [
            84, 78, 71, 80, 2, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0,
            0, 200, 1, 0, 0, 123, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 6, 8, 10,
            12, 5, 64, 0, 18, 38, 224, 186, 0, 0, 0, 14, 0, 0, 0, 20, 0, 0, 0, 206, 7, 0, 0, 2, 0,
            0, 0, 121, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 26,
            0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 128, 5, 0, 0, 2, 0, 0, 4, 0, 0, 33,
            66, 0, 2, 0, 0, 0, 26, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 64, 5, 0, 0,
            2, 0, 0, 2, 0, 0, 133, 8, 0, 2, 0, 0, 0, 26, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1,
            0, 0, 0, 64, 5, 0, 0, 2, 0, 0, 2, 0, 0, 133, 8, 0, 0, 0, 0, 6, 0, 0, 0, 0, 20, 0, 0, 0,
            24, 0, 8, 13, 0, 0, 0, 156, 11, 0, 0, 0, 10, 18, 4, 0, 0, 50, 141, 16, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(nitems, expected.len(), "nitems must match expected length");
        assert_eq!(&data[..nitems], expected, "bytewise equality");

        assert_eq!(u32le(&data[0..4]), MAGIC_INT_POS);
        assert_eq!(u32le(&data[4..8]), n_atoms);
        assert_eq!(u32le(&data[8..12]), n_frames);
        assert_eq!(u32le(&data[12..16]), initial_coding as u32);
        assert_eq!(u32le(&data[16..20]), initial_coding_parameter as u32);
        assert_eq!(u32le(&data[20..24]), coding as u32);
        assert_eq!(u32le(&data[24..28]), coding_parameter as u32);
        assert_eq!(u32le(&data[28..32]), prec_lo.into());
        assert_eq!(u32le(&data[32..36]), prec_hi.into());
    }

    #[test]
    fn xtc3_then_onetoone_remaining_xtc2() {
        let mut quant = [7, -7, 14, -14, 21, -21, -1, -2, -3, 10, 20, 30];
        let mut data = [0; 1024];
        let n_atoms = 2;
        let n_frames = 2;
        let speed = 6;
        let initial_coding = TNG_COMPRESS_ALGO_POS_XTC3;
        let initial_coding_parameter = 0;
        let coding = TNG_COMPRESS_ALGO_POS_XTC2;
        let coding_parameter = 0;
        let prec_hi = FixT::from(0);
        let prec_lo = FixT::from(0);
        let nitems = compress_quantized_pos(
            &mut quant,
            None,
            None,
            n_atoms,
            n_frames,
            speed,
            initial_coding,
            initial_coding_parameter,
            coding,
            coding_parameter,
            prec_hi,
            prec_lo,
            &mut Some(&mut data),
        );

        assert_eq!(u32le(&data[0..4]), MAGIC_INT_POS);
        assert_eq!(u32le(&data[4..8]), n_atoms);
        assert_eq!(u32le(&data[8..12]), n_frames);
        assert_eq!(u32le(&data[12..16]), initial_coding as u32);
        assert_eq!(u32le(&data[16..20]), initial_coding_parameter as u32);
        assert_eq!(u32le(&data[20..24]), coding as u32);
        assert_eq!(u32le(&data[24..28]), coding_parameter as u32);
        assert_eq!(u32le(&data[28..32]), prec_lo.into());
        assert_eq!(u32le(&data[32..36]), prec_hi.into());

        let expected = [
            84, 78, 71, 80, 2, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 187, 0, 0, 0, 28, 0, 0, 0, 14, 0, 0, 0, 42, 0, 0, 0, 2, 0,
            0, 0, 121, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 26,
            0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 128, 5, 0, 0, 2, 0, 0, 4, 0, 0, 33,
            66, 0, 2, 0, 0, 0, 26, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 64, 5, 0, 0,
            2, 0, 0, 2, 0, 0, 133, 8, 0, 2, 0, 0, 0, 26, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1,
            0, 0, 0, 64, 5, 0, 0, 2, 0, 0, 2, 0, 0, 133, 8, 0, 0, 0, 0, 6, 0, 0, 0, 0, 21, 0, 0, 0,
            24, 0, 8, 22, 0, 0, 0, 206, 1, 29, 0, 0, 0, 28, 0, 36, 0, 0, 0, 236, 4, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 6, 8, 10, 12, 5, 64,
            0, 18, 38, 224,
        ];
        assert_eq!(nitems, expected.len(), "nitems must match expected length");
        assert_eq!(&data[..nitems], expected, "bytewise equality");
    }

    #[test]
    fn bwlzh_intra_then_intra_remaining() {
        let mut quant = [1, 2, 3, 4, 5, 6, -6, -5, -4, -3, -2, -1];
        let mut quant_intra = quant;
        let mut data = [0; 1024];
        let n_atoms = 2;
        let n_frames = 2;
        let speed = 9;
        let initial_coding = TNG_COMPRESS_ALGO_POS_BWLZH_INTRA;
        let initial_coding_parameter = 0;
        let coding = TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA;
        let coding_parameter = 0;
        let prec_hi = FixT::from(42);
        let prec_lo = FixT::from(24);
        let nitems = compress_quantized_pos(
            &mut quant,
            None,
            Some(&mut quant_intra),
            n_atoms,
            n_frames,
            speed,
            initial_coding,
            initial_coding_parameter,
            coding,
            coding_parameter,
            prec_hi,
            prec_lo,
            &mut Some(&mut data),
        );

        assert_eq!(u32le(&data[0..4]), MAGIC_INT_POS);
        assert_eq!(u32le(&data[4..8]), n_atoms);
        assert_eq!(u32le(&data[8..12]), n_frames);
        assert_eq!(u32le(&data[12..16]), initial_coding as u32);
        assert_eq!(u32le(&data[16..20]), initial_coding_parameter as u32);
        assert_eq!(u32le(&data[20..24]), coding as u32);
        assert_eq!(u32le(&data[24..28]), coding_parameter as u32);
        assert_eq!(u32le(&data[28..32]), prec_lo.into());
        assert_eq!(u32le(&data[32..36]), prec_hi.into());

        let expected = [
            84, 78, 71, 80, 2, 0, 0, 0, 2, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
            24, 0, 0, 0, 42, 0, 0, 0, 131, 0, 0, 0, 255, 255, 255, 255, 6, 0, 0, 0, 6, 0, 0, 0, 6,
            0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 30, 0, 0, 0, 1, 0, 6, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0,
            0, 135, 120, 8, 0, 0, 5, 0, 0, 7, 0, 0, 17, 69, 28, 113, 0, 0, 3, 0, 0, 0, 27, 0, 0, 0,
            1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 88, 6, 0, 0, 3, 0, 0, 2, 0, 0, 134, 40, 128,
            0, 3, 0, 0, 0, 27, 0, 0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 88, 6, 0, 0, 3,
            0, 0, 2, 0, 0, 134, 40, 128, 8, 0, 0, 0, 0, 0, 0, 12, 242, 163, 100, 32,
        ];
        assert_eq!(nitems, expected.len(), "nitems must match expected length");
        assert_eq!(&data[..nitems], expected, "bytewise equality");
    }
}
