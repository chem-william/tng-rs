use std::{fmt::Display, ops::Mul};

use log::debug;

use crate::{
    coder::Coder,
    fix_point::{FixT, f64_to_fixt_pair, fixt_pair_to_f64},
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
pub(crate) const MAGIC_INT_POS: u32 = 0x50474E54;
pub(crate) const MAGIC_INT_VEL: u32 = 0x56474E54;

/// Default to relatively fast compression. For very good compression it makes sense
/// to choose speed = 4 or speed = 5.
const SPEED_DEFAULT: usize = 2;

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
    let mut quant = vec![0; total];

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

pub(crate) fn quantize(
    x: &[f64],
    n_atoms: usize,
    n_frames: usize,
    precision: f64,
) -> Result<Vec<i32>, ()> {
    let total = n_atoms
        .checked_mul(n_frames)
        .and_then(|v| v.checked_mul(3))
        .expect("overflow computing quant length");
    let mut quant = vec![0; total];

    for iframe in 0..n_frames {
        for i in 0..n_atoms {
            for j in 0..3 {
                quant[iframe * n_atoms * 3 + i * 3 + j] =
                    ((x[iframe * n_atoms * 3 + i * 3 + j] / precision) + 0.5).floor() as i32;
            }
        }
    }

    if verify_input_data(x, n_atoms, n_frames, precision).is_ok() {
        Ok(quant)
    } else {
        Err(())
    }
}

fn verify_input_data(x: &[f64], n_atoms: usize, n_frames: usize, precision: f64) -> Result<(), ()> {
    for iframe in 0..n_frames {
        for i in 0..n_atoms {
            for j in 0..3 {
                if (x[iframe * n_atoms * 3 + i * 3 + j] / precision + 0.5).abs()
                    >= f64::from(MAX_FVAL)
                {
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

pub(crate) trait Float: Copy + Mul<Output = Self> + Display {
    fn from_i32(v: i32) -> Self;
    fn from_f64(v: f64) -> Self;
}

impl Float for f64 {
    fn from_i32(v: i32) -> Self {
        v as f64
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl Float for f32 {
    fn from_i32(v: i32) -> Self {
        v as f32
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

pub(crate) fn unquantize<T: Float>(
    x: &mut [T],
    n_atoms: usize,
    n_frames: usize,
    precision: T,
    quant: &[i32],
) {
    for iframe in 0..n_frames {
        for i in 0..n_atoms {
            for j in 0..3 {
                x[iframe * n_atoms * 3 + i * 3 + j] =
                    T::from_i32(quant[iframe * n_atoms * 3 + i * 3 + j]) * precision;
            }
        }
    }
}

pub(crate) fn unquantize_inter_differences<T: Float>(
    x: &mut [T],
    n_atoms: usize,
    n_frames: usize,
    precision: T,
    quant: &[i32],
) {
    for i in 0..n_atoms {
        for j in 0..3 {
            let mut q = quant[i * 3 + j]; // First value
            x[i * 3 + j] = T::from_i32(q) * precision;
            for iframe in 1..n_frames {
                q += quant[iframe * n_atoms * 3 + i * 3 + j];
                x[iframe * n_atoms * 3 + i * 3 + j] = T::from_i32(q) * precision;
            }
        }
    }
}

pub(crate) fn unquantize_inter_differences_int(
    x: &mut [i32],
    n_atoms: usize,
    n_frames: usize,
    quant: &[i32],
) {
    for i in 0..n_atoms {
        for j in 0..3 {
            let mut q = quant[i * 3 + j]; // First value
            x[i * 3 + j] = q;
            for iframe in 1..n_frames {
                q += quant[iframe * n_atoms * 3 + i * 3 + j];
                x[iframe * n_atoms * 3 + i * 3 + j] = q;
            }
        }
    }
}

/// In frame update required for the initial frame intra-frame compression was used
pub(crate) fn unquantize_intra_differences_first_frame(quant: &mut [i32], natoms: usize) {
    for j in 0..3 {
        let mut q = quant[j];
        for i in 0..natoms {
            q += quant[i * 3 + j];
            quant[i * 3 + j] = q;
        }
    }
}

pub(crate) fn unquantize_intra_differences<T: Float>(
    x: &mut [T],
    n_atoms: usize,
    n_frames: usize,
    precision: T,
    quant: &[i32],
) {
    debug!("UQ precision={precision}");

    for iframe in 0..n_frames {
        for j in 0..3 {
            let mut q = quant[iframe * n_atoms * 3 + j];
            x[iframe * n_atoms * 3 + j] = T::from_i32(q) * precision;
            for i in 1..n_atoms {
                q += quant[iframe * n_atoms * 3 + i * 3 + j];
                x[iframe * n_atoms * 3 + i * 3 + j] = T::from_i32(q) * precision;
            }
        }
    }
}

pub(crate) fn unquantize_intra_differences_int(
    x: &mut [i32],
    n_atoms: usize,
    n_frames: usize,
    quant: &[i32],
) {
    for iframe in 0..n_frames {
        for j in 0..3 {
            let mut q = quant[iframe * n_atoms * 3 + j];
            x[iframe * n_atoms * 3 + j] = q;
            for i in 1..n_atoms {
                q += quant[iframe * n_atoms * 3 + i * 3 + j];
                x[iframe * n_atoms * 3 + i * 3 + j] = q;
            }
        }
    }
}

#[cfg(test)]
mod quantize_tests {
    use super::*;

    #[test]
    fn one_atom() {
        let x = vec![0.1, 0.2, 0.3];
        let quant = quantize(&x, 1, 1, 0.1).unwrap();
        let expected = [1, 2, 3];
        assert_eq!(quant, expected);
    }

    #[test]
    fn one_atom_two_frames() {
        let x = vec![0.05, 0.25, 0.55, 0.95, 1.25, 1.55];
        let quant = quantize(&x, 1, 2, 0.1).unwrap();
        let expected = [1, 3, 6, 9, 13, 16];
        assert_eq!(quant, expected);
    }

    #[test]
    fn two_atom_one_frame() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let quant = quantize(&x, 2, 1, 0.5).unwrap();
        let expected = [0, 2, 4, 6, 8, 10];
        assert_eq!(quant, expected);
    }

    #[test]
    fn two_atom_two_frames() {
        let x = vec![
            0.0, 0.25, 0.49, 1.0, 1.25, 1.49, 2.0, 2.25, 2.49, 3.0, 3.25, 3.49,
        ];
        let quant = quantize(&x, 2, 2, 0.5).unwrap();
        let expected = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7];
        assert_eq!(quant, expected);
    }
}

#[cfg(test)]
mod quant_tests {
    use super::*;

    #[test]
    fn one_atom_two_frames() {
        let quant = vec![1, 2, 3, 4, 5, 6];
        let quant_inter = quant_inter_differences(&quant, 1, 2);
        let quant_intra = quant_intra_differences(&quant, 1, 2);

        let expected_inter = [1, 2, 3, 3, 3, 3];
        assert_eq!(quant_inter, expected_inter);

        let expected_intra = [1, 2, 3, 4, 5, 6];
        assert_eq!(quant_intra, expected_intra);
    }

    #[test]
    fn two_atom_two_frames() {
        let quant = vec![
            1, 2, 3, 7, 8, 9, // frame 0
            2, 4, 6, 10, 12, 14, // frame 1
        ];
        let quant_inter = quant_inter_differences(&quant, 2, 2);
        let quant_intra = quant_intra_differences(&quant, 2, 2);

        let expected_inter = [1, 2, 3, 7, 8, 9, 1, 2, 3, 3, 4, 5];
        assert_eq!(quant_inter, expected_inter);

        let expected_intra = [1, 2, 3, 6, 6, 6, 2, 4, 6, 8, 8, 8];
        assert_eq!(quant_intra, expected_intra);
    }

    #[test]
    fn two_atom_three_frames() {
        let quant = vec![
            0, 0, 0, 1, 1, 1, // frame 0
            1, 2, 3, 2, 3, 4, // frame 1
            3, 5, 7, 4, 6, 8, // frame 2
        ];
        let quant_inter = quant_inter_differences(&quant, 2, 3);
        let quant_intra = quant_intra_differences(&quant, 2, 3);

        let expected_inter = [0, 0, 0, 1, 1, 1, 1, 2, 3, 1, 2, 3, 2, 3, 4, 2, 3, 4];
        assert_eq!(quant_inter, expected_inter);

        let expected_intra = [0, 0, 0, 1, 1, 1, 1, 2, 3, 1, 1, 1, 3, 5, 7, 1, 1, 1];
        assert_eq!(quant_intra, expected_intra);
    }
}

pub(crate) fn tng_compress_pos(
    pos: &[f64],
    n_atoms: usize,
    n_frames: usize,
    desired_precision: f64,
    speed: usize,
    algo: &mut [i32],
) -> Option<Vec<u8>> {
    let (prec_hi, prec_lo) = f64_to_fixt_pair(desired_precision);
    let quant = quantize(pos, n_atoms, n_frames, precision(prec_hi, prec_lo));
    if let Ok(mut ok_quant) = quant {
        Some(tng_compress_pos_int(
            &mut ok_quant,
            u32::try_from(n_atoms).expect("usize from u32"),
            u32::try_from(n_frames).expect("usize from u32"),
            prec_hi,
            prec_lo,
            speed,
            algo,
        ))
    } else {
        None
    }
}
pub(crate) fn tng_compress_pos_int(
    pos: &mut [i32],
    n_atoms: u32,
    n_frames: u32,
    prec_hi: FixT,
    prec_lo: FixT,
    speed: usize,
    algo: &mut [i32],
) -> Vec<u8> {
    // 12 bytes are required to store 4 32 bit integers
    // This is 17% extra. The final 11*4 is to store information needed for decompression
    let mut data =
        vec![0u8; usize::try_from(n_atoms * n_frames).expect("usize from u32") * 14 + 11 * 4];
    let quant = pos; // Already quantized positions
    let mut inner_speed = if speed == 0 { SPEED_DEFAULT } else { speed };

    // Boundaries of `speed`
    inner_speed = inner_speed.clamp(1, 6);

    let mut initial_coding = algo[0];
    let mut initial_coding_parameter = algo[1];
    let mut coding = algo[2];
    let mut coding_parameter = algo[3];

    let mut quant_inter = quant_inter_differences(
        quant,
        usize::try_from(n_atoms).expect("usize from u32"),
        usize::try_from(n_frames).expect("usize from u32"),
    );
    let mut quant_intra = quant_intra_differences(
        quant,
        usize::try_from(n_atoms).expect("usize from u32"),
        usize::try_from(n_frames).expect("usize from u32"),
    );

    // If any of the above codings / coding parameters are == -1, the optimal parameters must be found
    if initial_coding == -1 {
        initial_coding_parameter = -1;

        (initial_coding, initial_coding_parameter) = determine_best_pos_initial_coding(
            quant,
            &mut quant_intra,
            usize::try_from(n_atoms).expect("usize from u32"),
            inner_speed,
            prec_hi,
            prec_lo,
            initial_coding,
            initial_coding_parameter,
        );
    } else if initial_coding_parameter == -1 {
        (initial_coding, initial_coding_parameter) = determine_best_pos_initial_coding(
            quant,
            &mut quant_intra,
            usize::try_from(n_atoms).expect("usize from u32"),
            inner_speed,
            prec_hi,
            prec_lo,
            initial_coding,
            initial_coding_parameter,
        );
    }

    if n_frames == 1 {
        coding = 0;
        coding_parameter = 0;
    }

    if n_frames > 1 {
        if coding == -1 {
            coding_parameter = -1;
            determine_best_pos_coding(
                quant,
                &mut Some(&mut quant_inter),
                &mut Some(&mut quant_intra),
                n_atoms,
                n_frames,
                inner_speed,
                prec_hi,
                prec_lo,
                &mut coding,
                &mut coding_parameter,
            );
        } else if coding_parameter == -1 {
            determine_best_pos_coding(
                quant,
                &mut Some(&mut quant_inter),
                &mut Some(&mut quant_intra),
                n_atoms,
                n_frames,
                inner_speed,
                prec_hi,
                prec_lo,
                &mut coding,
                &mut coding_parameter,
            );
        }
    }

    let nitems = compress_quantized_pos(
        quant,
        Some(&mut quant_inter),
        Some(&mut quant_intra),
        n_atoms,
        n_frames,
        inner_speed,
        initial_coding,
        initial_coding_parameter,
        coding,
        coding_parameter,
        prec_hi,
        prec_lo,
        &mut Some(&mut data),
    );
    data.truncate(nitems);

    if algo[0] == -1 {
        algo[0] = initial_coding;
    }

    if algo[1] == -1 {
        algo[1] = initial_coding_parameter;
    }

    if algo[2] == -1 {
        algo[2] = coding;
    }

    if algo[3] == -1 {
        algo[3] = coding_parameter;
    }

    data
}

pub(crate) fn tng_compress_pos_float(
    pos: &[f32],
    n_atoms: usize,
    n_frames: usize,
    desired_precision: f32,
    speed: usize,
    algo: &mut [i32],
) -> Option<Vec<u8>> {
    let (prec_hi, prec_lo) = f64_to_fixt_pair(f64::from(desired_precision));
    let quant = quantize_float(pos, n_atoms, n_frames, precision(prec_hi, prec_lo) as f32);

    if let Ok(mut ok_quant) = quant {
        Some(tng_compress_pos_int(
            &mut ok_quant,
            u32::try_from(n_atoms).expect("usize from u32"),
            u32::try_from(n_frames).expect("usize from u32"),
            prec_hi,
            prec_lo,
            speed,
            algo,
        ))
    } else {
        None
    }
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
                n_atoms.try_into().expect("u32 from usize"),
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
                n_atoms.try_into().expect("u32 from usize"),
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

pub(crate) fn readbufferfix(buf: &[u8], num: i32) -> FixT {
    let mut num = num;

    let mut cnt = 0;
    let mut b: u8;
    let mut shift = 0;
    let mut f = FixT::from(0);
    loop {
        b = buf[cnt];
        cnt += 1;
        f |= (FixT::from(b as u32) & FixT::from(0xFF)) << FixT::from(shift);
        shift += 8;

        num -= 1;
        if num == 0 {
            break;
        }
    }
    f
}

/// Perform position compression from the quantized data
pub(crate) fn compress_quantized_pos(
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
        println!("{:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;

    // Number of atoms
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], FixT::from(n_atoms), 4);
        println!("{:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;

    // Number of frames
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], FixT::from(n_frames), 4);
        println!("{:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;

    // Initial coding
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(initial_coding).expect("u32 from i32")),
            4,
        );
        println!("{:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;

    // Initial coding parameter
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(initial_coding_parameter).expect("u32 from i32")),
            4,
        );
        println!("{:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;

    // Coding
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(coding).expect("u32 from i32")),
            4,
        );
        println!("{:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;

    // Coding parameter
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(coding_parameter).expect("u32 from i32")),
            4,
        );
        println!("{:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;

    // Precision
    println!("{prec_lo}");
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], prec_lo, 4);
        println!("Precision lo: {:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;
    if let Some(mut_data) = data.as_mut() {
        bufferfix(&mut mut_data[bufloc..], prec_hi, 4);
        println!("Precision hi: {:?}", &mut_data[bufloc..bufloc + 4]);
    }
    bufloc += 4;

    println!("printing intermediate data");
    println!("{:?}", data);
    println!("======================");

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
            println!("datablock: {:?}", datablock);
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
        println!("we skip here");
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
    println!("bufloc: {bufloc}");
    bufloc
}

pub(crate) fn determine_best_pos_coding(
    quant: &mut [i32],
    quant_inter: &mut Option<&mut [i32]>,
    quant_intra: &mut Option<&mut [i32]>,
    n_atoms: u32,
    n_frames: u32,
    speed: usize,
    prec_hi: FixT,
    prec_lo: FixT,
    coding: &mut i32,
    coding_parameter: &mut i32,
) {
    if *coding == -1 {
        // Determine all parameters automatically
        let mut best_coding: i32;
        let mut best_coding_parameter: i32;
        let mut current_coding: i32;
        let mut current_coding_parameter: i32;
        let mut current_code_size: i32;

        // Always use XTC2 for the initial coding
        let initial_code_size = i32::try_from(compress_quantized_pos(
            quant,
            quant_inter.as_deref_mut(),
            quant_intra.as_deref_mut(),
            n_atoms,
            1,
            speed,
            TNG_COMPRESS_ALGO_POS_XTC2,
            0,
            0,
            0,
            prec_hi,
            prec_lo,
            &mut None,
        ))
        .expect("i32 from usize");
        // Start with XTC2, it should always work
        current_coding = TNG_COMPRESS_ALGO_POS_XTC2;
        current_coding_parameter = 0;
        let mut best_code_size = i32::try_from(compress_quantized_pos(
            quant,
            quant_inter.as_deref_mut(),
            quant_intra.as_deref_mut(),
            n_atoms,
            n_frames,
            speed,
            TNG_COMPRESS_ALGO_POS_XTC2,
            0,
            current_coding,
            current_coding_parameter,
            prec_hi,
            prec_lo,
            &mut None,
        ))
        .expect("i32 from usize");
        best_coding = current_coding;
        best_coding_parameter = current_coding_parameter;
        best_code_size -= initial_code_size; // Correcet for the use of XTC2 for the first frame

        // Determine best parameter for stopbit interframe coding
        current_coding = TNG_COMPRESS_ALGO_POS_STOPBIT_INTER;
        let mut coder = Coder::default();
        let mut current_code_size_usize: usize = (n_atoms * 3 * (n_frames - 1))
            .try_into()
            .expect("usize from u32");
        if !coder.determine_best_coding_stop_bits(
            &mut quant_inter.as_mut().expect("quant_inter to be Some")
                [(n_atoms * 3).try_into().expect("u32 to usize")..],
            &mut current_code_size_usize,
            &mut current_coding_parameter,
            n_atoms.try_into().expect("usize from u32"),
        ) {
            current_code_size = i32::try_from(current_code_size_usize).expect("i32 from usize");
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
                best_code_size = current_code_size;
            }
        }

        // Determine best parameter for triplet interframe coding
        current_coding = TNG_COMPRESS_ALGO_POS_TRIPLET_INTER;
        coder = Coder::default();
        current_code_size_usize = (n_atoms * 3 * (n_frames - 1))
            .try_into()
            .expect("usize from u32");
        current_coding_parameter = 0;
        if !coder.determine_best_coding_triple(
            &mut quant_inter.as_mut().expect("quant_inter to be Some")
                [(n_atoms * 3).try_into().expect("u32 to usize")..],
            &mut current_code_size_usize,
            &mut current_coding_parameter,
            n_atoms.try_into().expect("usize from u32"),
        ) {
            current_code_size = i32::try_from(current_code_size_usize).expect("i32 from usize");
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
                best_code_size = current_code_size;
            }
        }

        // Determine best parameter for triplet intraframe coding
        current_coding = TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA;
        coder = Coder::default();
        current_code_size_usize = (n_atoms * 3 * (n_frames - 1))
            .try_into()
            .expect("usize from u32");
        current_coding_parameter = 0;
        if !coder.determine_best_coding_triple(
            &mut quant_intra.as_mut().expect("quant_intra to be Some")
                [(n_atoms * 3).try_into().expect("u32 to usize")..],
            &mut current_code_size_usize,
            &mut current_coding_parameter,
            n_atoms.try_into().expect("usize from u32"),
        ) {
            current_code_size = i32::try_from(current_code_size_usize).expect("i32 from usize");
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
                best_code_size = current_code_size;
            }
        }

        // Determine best parameter for triplet one-to-one coding
        current_coding = TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE;
        coder = Coder::default();
        current_code_size_usize = (n_atoms * 3 * (n_frames - 1))
            .try_into()
            .expect("usize from u32");
        current_coding_parameter = 0;
        if !coder.determine_best_coding_triple(
            &mut quant[(n_atoms * 3).try_into().expect("u32 to usize")..],
            &mut current_code_size_usize,
            &mut current_coding_parameter,
            n_atoms.try_into().expect("usize from u32"),
        ) {
            current_code_size = i32::try_from(current_code_size_usize).expect("i32 from usize");
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
                best_code_size = current_code_size;
            }
        }

        // Test BWLZH inter
        if speed >= 4 {
            current_coding = TNG_COMPRESS_ALGO_POS_BWLZH_INTER;
            current_coding_parameter = 0;
            current_code_size = i32::try_from(compress_quantized_pos(
                quant,
                quant_inter.as_deref_mut(),
                quant_intra.as_deref_mut(),
                n_atoms,
                n_frames,
                speed,
                TNG_COMPRESS_ALGO_POS_XTC2,
                0,
                current_coding,
                current_coding_parameter,
                prec_hi,
                prec_lo,
                &mut None,
            ))
            .expect("i32 from usize");
            current_code_size -= initial_code_size; // Correct for the use of XTC2 for the first time
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
            current_code_size = i32::try_from(compress_quantized_pos(
                quant,
                quant_inter.as_deref_mut(),
                quant_intra.as_deref_mut(),
                n_atoms,
                n_frames,
                speed,
                TNG_COMPRESS_ALGO_POS_XTC2,
                0,
                current_coding,
                current_coding_parameter,
                prec_hi,
                prec_lo,
                &mut None,
            ))
            .expect("i32 from usize");
            current_code_size -= initial_code_size; // Correct for the use of XTC2 for the first time
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
            }
        }
        *coding = best_coding;
        *coding_parameter = best_coding_parameter;
    } else if *coding_parameter == -1 {
        let unpacked_quant_inter = quant_inter.as_mut().expect("quant_inter to be Some");
        match *coding {
            TNG_COMPRESS_ALGO_POS_XTC2
            | TNG_COMPRESS_ALGO_POS_XTC3
            | TNG_COMPRESS_ALGO_POS_BWLZH_INTER
            | TNG_COMPRESS_ALGO_POS_BWLZH_INTRA => {
                *coding_parameter = 0;
            }
            TNG_COMPRESS_ALGO_POS_STOPBIT_INTER => {
                let mut coder = Coder::default();
                let current_code_size =
                    i32::try_from(n_atoms * 3 * (n_frames - 1)).expect("i32 from u32");
                coder.determine_best_coding_stop_bits(
                    &mut unpacked_quant_inter[(n_atoms * 3).try_into().expect("u32 to usize")..],
                    &mut current_code_size.try_into().expect("into u32"),
                    coding_parameter,
                    n_atoms.try_into().expect("usize from u32"),
                );
            }
            TNG_COMPRESS_ALGO_POS_TRIPLET_INTER => {
                let mut coder = Coder::default();
                let current_code_size =
                    i32::try_from(n_atoms * 3 * (n_frames - 1)).expect("i32 from u32");
                coder.determine_best_coding_triple(
                    &mut unpacked_quant_inter[(n_atoms * 3).try_into().expect("u32 to usize")..],
                    &mut current_code_size.try_into().expect("into u32"),
                    coding_parameter,
                    n_atoms.try_into().expect("usize from u32"),
                );
            }
            TNG_COMPRESS_ALGO_POS_TRIPLET_INTRA => {
                let mut coder = Coder::default();
                let current_code_size =
                    i32::try_from(n_atoms * 3 * (n_frames - 1)).expect("i32 from u32");
                coder.determine_best_coding_triple(
                    &mut quant_intra.as_mut().expect("quant_intra to be Some")
                        [(n_atoms * 3).try_into().expect("u32 to usize")..],
                    &mut current_code_size.try_into().expect("into u32"),
                    coding_parameter,
                    n_atoms.try_into().expect("usize from u32"),
                );
            }
            TNG_COMPRESS_ALGO_POS_TRIPLET_ONETOONE => {
                let mut coder = Coder::default();
                let current_code_size =
                    i32::try_from(n_atoms * 3 * (n_frames - 1)).expect("i32 from u32");
                coder.determine_best_coding_triple(
                    &mut quant[(n_atoms * 3).try_into().expect("u32 to usize")..],
                    &mut current_code_size.try_into().expect("into u32"),
                    coding_parameter,
                    n_atoms.try_into().expect("usize from u32"),
                );
            }
            _ => {}
        }
    }
}

pub(crate) fn tng_compress_vel(
    vel: &[f64],
    n_atoms: usize,
    n_frames: usize,
    desired_precision: f64,
    speed: usize,
    algo: &mut [i32],
) -> Option<Vec<u8>> {
    let (prec_hi, prec_lo) = f64_to_fixt_pair(desired_precision);
    let quant = quantize(vel, n_atoms, n_frames, precision(prec_hi, prec_lo));
    if let Ok(mut ok_quant) = quant {
        Some(tng_compress_vel_int(
            &mut ok_quant,
            n_atoms,
            n_frames,
            prec_hi,
            prec_lo,
            speed,
            algo,
        ))
    } else {
        None
    }
}

pub(crate) fn tng_compress_vel_float(
    vel: &[f32],
    n_atoms: usize,
    n_frames: usize,
    desired_precision: f32,
    speed: usize,
    algo: &mut [i32],
) -> Option<Vec<u8>> {
    let (prec_hi, prec_lo) = f64_to_fixt_pair(f64::from(desired_precision));
    let quant = quantize_float(vel, n_atoms, n_frames, precision(prec_hi, prec_lo) as f32);

    if let Ok(mut ok_quant) = quant {
        Some(tng_compress_vel_int(
            &mut ok_quant,
            n_atoms,
            n_frames,
            prec_hi,
            prec_lo,
            speed,
            algo,
        ))
    } else {
        None
    }
}

pub(crate) fn tng_compress_vel_int(
    vel: &mut [i32],
    n_atoms: usize,
    n_frames: usize,
    prec_hi: FixT,
    prec_lo: FixT,
    speed: usize,
    algo: &mut [i32],
) -> Vec<u8> {
    // 12 bytes are required to store 4 32 bit integers
    // This is 17% extra. The final 11*4 is to store information needed for decompression
    let mut data = vec![0u8; n_atoms * n_frames * 14 + 11 * 4];
    let quant = vel;
    let mut inner_speed = if speed == 0 { SPEED_DEFAULT } else { speed };

    // Boundaries of `speed`
    inner_speed = inner_speed.clamp(1, 6);

    let mut initial_coding = algo[0];
    let mut initial_coding_parameter = algo[1];
    let mut coding = algo[2];
    let mut coding_parameter = algo[3];

    let mut quant_inter = quant_inter_differences(quant, n_atoms, n_frames);

    // If any of the above codings / coding parameters are == -1, the optimal parameters must be found
    if initial_coding == -1 {
        initial_coding_parameter = -1;

        (initial_coding, initial_coding_parameter) = determine_best_vel_initial_coding(
            quant,
            n_atoms,
            inner_speed,
            prec_hi,
            prec_lo,
            initial_coding,
            initial_coding_parameter,
        );
    } else if initial_coding_parameter == -1 {
        (initial_coding, initial_coding_parameter) = determine_best_vel_initial_coding(
            quant,
            n_atoms,
            inner_speed,
            prec_hi,
            prec_lo,
            initial_coding,
            initial_coding_parameter,
        );
    }

    if n_frames == 1 {
        coding = 0;
        coding_parameter = 0;
    }

    if n_frames > 1 {
        if coding == -1 {
            coding_parameter = -1;
            determine_best_vel_coding(
                quant,
                &mut Some(&mut quant_inter),
                n_atoms,
                n_frames,
                inner_speed,
                prec_hi,
                prec_lo,
                coding,
                coding_parameter,
            );
        } else if coding_parameter == -1 {
            determine_best_vel_coding(
                quant,
                &mut Some(&mut quant_inter),
                n_atoms,
                n_frames,
                inner_speed,
                prec_hi,
                prec_lo,
                coding,
                coding_parameter,
            );
        }
    }
    let nitems = compress_quantized_vel(
        quant,
        Some(&mut quant_inter),
        n_atoms,
        n_frames,
        inner_speed,
        initial_coding,
        initial_coding_parameter,
        coding,
        coding_parameter,
        prec_hi,
        prec_lo,
        &mut Some(&mut data),
    );
    data.truncate(nitems);

    if algo[0] == -1 {
        algo[0] = initial_coding;
    }

    if algo[1] == -1 {
        algo[1] = initial_coding_parameter;
    }

    if algo[2] == -1 {
        algo[2] = coding;
    }

    if algo[3] == -1 {
        algo[3] = coding_parameter;
    }

    data
}

fn determine_best_vel_coding(
    quant: &mut [i32],
    quant_inter: &mut Option<&mut [i32]>,
    n_atoms: usize,
    n_frames: usize,
    speed: usize,
    prec_hi: FixT,
    prec_lo: FixT,
    coding: i32,
    coding_parameter: i32,
) -> (i32, i32) {
    let mut resulting_coding = coding;
    let mut resulting_coding_parameter = coding_parameter;

    if coding == -1 {
        //Determine all parameters automatically
        let mut best_coding;
        let mut best_coding_parameter;
        let mut best_code_size;
        let mut current_coding;
        let mut current_coding_parameter;
        let mut current_code_size;
        let initial_numbits = 5;

        // Use stopbits one-to-one coding for the initial coding.
        let initial_code_size = compress_quantized_vel(
            quant,
            quant_inter.as_deref_mut(),
            n_atoms,
            1,
            speed,
            TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE,
            initial_numbits,
            0,
            0,
            prec_hi,
            prec_lo,
            &mut None,
        );

        // Test stopbit one-to-one
        current_coding = TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE;
        current_code_size = n_atoms * 3 * (n_frames - 1);
        current_coding_parameter = 00;
        let mut coder = Coder::default();
        coder.determine_best_coding_stop_bits(
            &mut quant[n_atoms * 3..],
            &mut current_code_size,
            &mut current_coding_parameter,
            n_atoms,
        );
        best_coding = current_coding;
        best_code_size = current_code_size;
        best_coding_parameter = current_coding_parameter;

        // Test triplet interframe
        current_coding = TNG_COMPRESS_ALGO_POS_TRIPLET_INTER;
        current_code_size = n_atoms * 3 * (n_frames - 1);
        current_coding_parameter = 0;
        coder = Coder::default();
        if !coder.determine_best_coding_triple(
            &mut quant_inter.as_mut().expect("quant_inter to be Some")[n_atoms * 3..],
            &mut current_code_size,
            &mut current_coding_parameter,
            n_atoms,
        ) && current_code_size < best_code_size
        {
            best_coding = current_coding;
            best_coding_parameter = current_coding_parameter;
            best_code_size = current_code_size;
        }

        // Test triplet one-to-one
        current_coding = TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE;
        current_code_size = n_atoms * 3 * (n_frames - 1);
        current_coding_parameter = 0;
        coder = Coder::default();
        if !coder.determine_best_coding_triple(
            &mut quant[n_atoms * 3..],
            &mut current_code_size,
            &mut current_coding_parameter,
            n_atoms,
        ) && current_code_size < best_code_size
        {
            best_coding = current_coding;
            best_code_size = current_code_size;
            best_coding_parameter = current_coding_parameter;
        }

        // Test stopbit interframe
        current_coding = TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER;
        current_code_size = n_atoms * 3 * (n_frames - 1);
        current_coding_parameter = 0;
        coder = Coder::default();
        if !coder.determine_best_coding_stop_bits(
            &mut quant_inter.as_mut().expect("quant_inter to be Some")[n_atoms * 3..],
            &mut current_code_size,
            &mut current_coding_parameter,
            n_atoms,
        ) && current_code_size < best_code_size
        {
            best_coding = current_coding;
            best_code_size = current_code_size;
            best_coding_parameter = current_coding_parameter;
        }

        if speed >= 4 {
            // Test BWLZH inter
            current_coding = TNG_COMPRESS_ALGO_VEL_BWLZH_INTER;
            current_coding_parameter = 0;
            current_code_size = compress_quantized_vel(
                quant,
                quant_inter.as_deref_mut(),
                n_atoms,
                n_frames,
                speed,
                TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE,
                initial_numbits,
                current_coding,
                current_coding_parameter,
                prec_hi,
                prec_lo,
                &mut None,
            );
            current_code_size -= initial_code_size; // Correct for the initial frame
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_code_size = current_code_size;
                best_coding_parameter = current_coding_parameter;
            }

            // Test BWLZH one-to-one
            current_coding = TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE;
            current_coding_parameter = 0;
            current_code_size = compress_quantized_vel(
                quant,
                quant_inter.as_deref_mut(),
                n_atoms,
                n_frames,
                speed,
                TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE,
                initial_numbits,
                current_coding,
                current_coding_parameter,
                prec_hi,
                prec_lo,
                &mut None,
            );
            current_code_size -= initial_code_size; // Correct for the initial frame
            if current_code_size < best_code_size {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
            }
        }
        resulting_coding = best_coding;
        resulting_coding_parameter = best_coding_parameter;
    } else if coding_parameter == -1 {
        if coding == TNG_COMPRESS_ALGO_VEL_BWLZH_INTER
            || coding == TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE
        {
            resulting_coding_parameter = 0;
        } else if coding == TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE {
            let mut coder = Coder::default();
            let mut current_code_size = n_atoms * 3 * (n_frames - 1);
            coder.determine_best_coding_stop_bits(
                &mut quant[n_atoms * 3..],
                &mut current_code_size,
                &mut resulting_coding_parameter,
                n_atoms,
            );
        } else if coding == TNG_COMPRESS_ALGO_VEL_TRIPLET_INTER {
            let mut coder = Coder::default();
            let mut current_code_size = n_atoms * 3 * (n_frames - 1);
            coder.determine_best_coding_triple(
                &mut quant_inter.as_deref_mut().expect("quant_inter to be Some")[n_atoms * 3..],
                &mut current_code_size,
                &mut resulting_coding_parameter,
                n_atoms,
            );
        } else if coding == TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE {
            let mut coder = Coder::default();
            let mut current_code_size = n_atoms * 3 * (n_frames - 1);
            coder.determine_best_coding_triple(
                &mut quant[n_atoms * 3..],
                &mut current_code_size,
                &mut resulting_coding_parameter,
                n_atoms,
            );
        } else if coding == TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER {
            let mut coder = Coder::default();
            let mut current_code_size = n_atoms * 3 * (n_frames - 1);
            coder.determine_best_coding_stop_bits(
                &mut quant_inter.as_deref_mut().expect("quant_inter to be Some")[n_atoms * 3..],
                &mut current_code_size,
                &mut resulting_coding_parameter,
                n_atoms,
            );
        }
    }
    (resulting_coding, resulting_coding_parameter)
}

/// Perform velocity compression from vel into the data block
fn compress_quantized_vel(
    quant: &mut [i32],
    quant_inter: Option<&mut [i32]>,
    n_atoms: usize,
    n_frames: usize,
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
        bufferfix(&mut mut_data[bufloc..], FixT::from(MAGIC_INT_VEL), 4);
    }
    bufloc += 4;

    // Number of atoms
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(n_atoms).expect("u32 from usize")),
            4,
        );
    }
    bufloc += 4;

    // Number of frames
    if let Some(mut_data) = data.as_mut() {
        bufferfix(
            &mut mut_data[bufloc..],
            FixT::from(u32::try_from(n_frames).expect("u32 from usize")),
            4,
        );
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
    let mut length = n_atoms * 3;
    match initial_coding {
        TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE
        | TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE
        | TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE => {
            let mut coder = Coder::default();
            let out_datablock;
            (out_datablock, output_length) = coder
                .pack_array(
                    quant,
                    &mut length,
                    initial_coding,
                    initial_coding_parameter,
                    n_atoms,
                    &mut speed,
                )
                .expect("packed array");
            datablock = Some(out_datablock);
        }
        _ => unreachable!(),
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
    if let Some(mut_data) = data.as_mut()
        && let Some(db) = datablock
    {
        mut_data[bufloc..bufloc + output_length].copy_from_slice(&db[..output_length]);
        bufloc += output_length;
    }

    if n_frames > 1 {
        // datablock = None;
        let us_natoms = n_atoms;
        let mut fallback_len = us_natoms
            .checked_mul(3)
            .and_then(|v| v.checked_mul(n_frames - 1))
            .expect("fallback_len overflow");
        let result = match coding {
            // Inter-frame compression?
            TNG_COMPRESS_ALGO_VEL_TRIPLET_INTER
            | TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER
            | TNG_COMPRESS_ALGO_VEL_BWLZH_INTER => {
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
            TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE
            | TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE
            | TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE => {
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

pub(crate) fn determine_best_vel_initial_coding(
    quant: &mut [i32],
    n_atoms: usize,
    speed: usize,
    prec_hi: FixT,
    prec_lo: FixT,
    initial_coding: i32,
    initial_coding_parameter: i32,
) -> (i32, i32) {
    let mut resulting_coding = initial_coding;
    let mut resulting_coding_parameter = initial_coding_parameter;
    if initial_coding == -1 {
        let mut best_coding = -1;
        let mut best_coding_parameter = -1;
        let mut best_code_size = -1;
        let mut current_coding;
        let mut current_coding_parameter;
        let mut current_code_size;
        let mut coder;
        // Start to determine best parameter for stopbit one-to-one
        current_coding = TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE;
        current_code_size = n_atoms * 3;
        current_coding_parameter = 0;
        coder = Coder::default();

        if !coder.determine_best_coding_stop_bits(
            quant,
            &mut current_code_size,
            &mut current_coding_parameter,
            n_atoms,
        ) {
            best_coding = current_coding;
            best_coding_parameter = current_coding_parameter;
            best_code_size = i32::try_from(current_code_size).expect("i32 from usize");
        }

        //Determine best parameter for triplet one-to-one
        current_coding = TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE;
        coder = Coder::default();
        current_code_size = n_atoms * 3;
        current_coding_parameter = 0;
        if !coder.determine_best_coding_triple(
            quant,
            &mut current_code_size,
            &mut current_coding_parameter,
            n_atoms,
        ) && (best_coding == -1
            || (i32::try_from(current_code_size).expect("i32 from usize") < best_code_size))
        {
            best_coding = current_coding;
            best_coding_parameter = current_coding_parameter;
            best_code_size = i32::try_from(current_code_size).expect("i32 from usize");
        }

        // Test BWLZH one-to-one
        if speed >= 4 {
            current_coding = TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE;
            current_coding_parameter = 0;
            current_code_size = compress_quantized_vel(
                quant,
                None,
                n_atoms,
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
            if (best_coding == -1)
                || (i32::try_from(current_code_size).expect("i32 from usize") < best_code_size)
            {
                best_coding = current_coding;
                best_coding_parameter = current_coding_parameter;
            }
        }
        resulting_coding = best_coding;
        resulting_coding_parameter = best_coding_parameter;
    } else if initial_coding_parameter == -1 {
        if initial_coding == TNG_COMPRESS_ALGO_VEL_BWLZH_ONETOONE {
            resulting_coding_parameter = 0;
        } else if initial_coding == TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE {
            let mut coder = Coder::default();
            let mut current_code_size = n_atoms * 3;
            coder.determine_best_coding_triple(
                quant,
                &mut current_code_size,
                &mut resulting_coding_parameter,
                n_atoms,
            );
        } else if initial_coding == TNG_COMPRESS_ALGO_VEL_TRIPLET_ONETOONE {
            let mut coder = Coder::default();
            let mut current_code_size = n_atoms * 3;
            coder.determine_best_coding_triple(
                quant,
                &mut current_code_size,
                &mut resulting_coding_parameter,
                n_atoms,
            );
        } else {
            unreachable!("initial_coding != -1, but initial_coding_parameter != -1")
        }
    };

    (resulting_coding, resulting_coding_parameter)
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
mod compress_quantized_pos {
    use super::*;

    fn u32le(b: &[u8]) -> u32 {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    #[test]
    fn compress_pos() {
        let pos = [0.0, 0.0, 0.0];
        let mut algo = [-1, -1, -1, -1];
        let _nitmes = -1;

        let result = tng_compress_pos(&pos, 1, 1, 1e-6, 1, &mut algo).unwrap();
        println!("{:?}", &result[..45]);
        println!("{:?}", result.len());
        let _expected = [
            84, 78, 71, 80, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -58, 16, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        // assert_eq!(result, expected);
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
