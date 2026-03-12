use std::ffi::c_int;
use std::sync::Once;

static INIT: Once = Once::new();

fn init_logger() {
    INIT.call_once(|| {
        env_logger::builder().is_test(true).try_init().ok();
    });
}

use crate::coder::Coder;
use crate::compress::{
    TNG_COMPRESS_ALGO_BWLZH1, TNG_COMPRESS_ALGO_BWLZH2, TNG_COMPRESS_ALGO_POS_XTC2,
    TNG_COMPRESS_ALGO_POS_XTC3, TNG_COMPRESS_ALGO_TRIPLET,
};
use crate::ffi;

fn rust_run_pack(input: &[i32], algo: i32, natoms: usize, speed: usize) -> (Vec<u8>, usize) {
    init_logger();
    let mut coder = Coder::default();
    let mut input = input.to_vec();
    let mut length = input.len();
    let mut speed = speed;
    let coding_param = 0;
    coder
        .pack_array(
            &mut input,
            &mut length,
            algo,
            coding_param,
            natoms,
            &mut speed,
        )
        .unwrap()
}

fn c_run_pack(input: &mut [i32], algo: c_int, natoms: i32, speed: c_int) -> (Vec<u8>, i32) {
    let c_coder = unsafe { ffi::Ptngc_coder_init() };
    let mut c_length = input.len() as c_int;
    let c_raw_output = unsafe {
        ffi::Ptngc_pack_array(
            c_coder,
            input.as_mut_ptr(),
            &mut c_length,
            algo,
            0 as c_int,
            natoms,
            speed,
        )
    };
    assert!(!c_raw_output.is_null(), "Ptngc_pack_array returned null");
    let mut c_output = vec![0; c_length as usize];
    unsafe { c_raw_output.copy_to(c_output.as_mut_ptr(), c_length as usize) };
    unsafe { libc::free(c_raw_output as *mut _) };
    unsafe { ffi::Ptngc_coder_deinit(c_coder) };

    (c_output, c_length)
}

#[test]
fn bwlzh1() {
    let input = vec![5, -10, 42, -7, 0, -999];

    let algo = TNG_COMPRESS_ALGO_BWLZH1;
    let (rust_output, rust_output_length) = rust_run_pack(&input, algo, 2, 9);
    let (c_output, _c_length) = c_run_pack(&mut input.clone(), algo, 2, 9);

    assert_eq!(rust_output[..rust_output_length], c_output);
}

#[test]
fn bwlzh2() {
    let input = vec![-1, -2, -3, 10, 20, 30];
    let algo = TNG_COMPRESS_ALGO_BWLZH2;
    let (rust_output, rust_output_length) = rust_run_pack(&input, algo, 1, 9);
    let (c_output, _c_length) = c_run_pack(&mut input.clone(), algo, 1, 9);

    assert_eq!(rust_output[..rust_output_length], c_output);
}

#[test]
fn xtc2() {
    let input = vec![-1, -2, -3, 10, 20, 30];
    let algo = TNG_COMPRESS_ALGO_POS_XTC2;
    let (rust_output, rust_output_length) = rust_run_pack(&input, algo, 1, 9);
    let (c_output, _c_length) = c_run_pack(&mut input.clone(), algo, 1, 9);

    assert_eq!(rust_output[..rust_output_length], c_output);
}

#[test]
fn xtc3() {
    let input = vec![-1, -2, -3, 10, 20, 30];
    let algo = TNG_COMPRESS_ALGO_POS_XTC3;
    let (rust_output, rust_output_length) = rust_run_pack(&input, algo, 1, 9);
    let (c_output, _c_length) = c_run_pack(&mut input.clone(), algo, 1, 9);

    assert_eq!(rust_output[..rust_output_length], c_output);
}

#[test]
fn triplet() {
    let input = vec![-1, -2, -3, 10, 20, 30];
    let algo = TNG_COMPRESS_ALGO_TRIPLET;
    let (rust_output, rust_output_length) = rust_run_pack(&input, algo, 1, 9);
    let (c_output, _c_length) = c_run_pack(&mut input.clone(), algo, 1, 9);

    assert_eq!(rust_output[..rust_output_length], c_output);
}

#[test]
fn bwlzh1_low_speed() {
    let input = vec![5, -10, 42, -7, 0, -999];
    let algo = TNG_COMPRESS_ALGO_BWLZH1;
    let (rust_output, rust_output_length) = rust_run_pack(&input, algo, 2, 4);
    let (c_output, _) = c_run_pack(&mut input.clone(), algo, 2, 4);
    assert_eq!(rust_output[..rust_output_length], c_output);
}

#[test]
fn bwlzh2_low_speed() {
    let input = vec![-1, -2, -3, 10, 20, 30];
    let algo = TNG_COMPRESS_ALGO_BWLZH2;
    let (rust_output, rust_output_length) = rust_run_pack(&input, algo, 1, 4);
    let (c_output, _) = c_run_pack(&mut input.clone(), algo, 1, 4);
    assert_eq!(rust_output[..rust_output_length], c_output);
}
