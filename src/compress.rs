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
    let total = (n_atoms as usize)
        .checked_mul(n_frames as usize)
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
    quant: &[i32],
    quant_intra: &[i32],
    n_atoms: usize,
    speed: usize,
    prec_hi: FixT,
    prec_lo: FixT,
    initial_coding: i32,
    initial_coding_parameter: i32,
) -> (i32, i32) {
    if initial_coding == -1 {
        let mut best_coding = 0;
        let mut best_coding_parameter = 0;
        let mut best_coding_size = 0;
        let mut current_coding: i32 = 0;
        let mut current_coding_parameter = 0;
        let mut current_coding_size = 0;
        let mut coder = Coder::default();

        // Start with XTC2, it should always work
        current_coding = TNG_COMPRESS_ALGO_POS_XTC2;
        current_coding_parameter = 0;
        // let nitems = compress_quantized_pos(
        //     quant,
        //     None,
        //     Some(quant_intra),
        //     n_atoms,
        //     1,
        //     speed,
        //     current_coding,
        //     current_coding_parameter,
        //     0,
        //     0,
        //     prec_hi,
        //     prec_lo,
        //     None,
        // );
    }
    (0, 0)
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

fn compress_quantized_pos(
    quant: &[i32],
    quant_inter: Option<&[i32]>,
    quant_intra: Option<&[i32]>,
    n_atoms: u32,
    n_frames: u32,
    speed: usize,
    initial_coding: i32,
    initial_coding_parameter: i32,
    coding: i32,
    coding_parameter: i32,
    prec_hi: FixT,
    prec_lo: FixT,
    data: &mut Option<&mut [u8]>,
) -> usize {
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
    match initial_coding {
        TNG_COMPRESS_ALGO_POS_XTC2
        | TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE
        | TNG_COMPRESS_ALGO_POS_XTC3 => {
            let coder = Coder::default();
            let length = n_atoms * 3;
            // let datablock =
            //     coder.pack_array(quant, length, coding, coding_parameter, n_atoms, speed);
        }
        TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA | TNG_COMPRESS_ALGO_POS_BWLZH_INTRA => {}
        _ => {}
    }

    0
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
