use std::ffi::{c_int, c_ulong};

use crate::compress;
use crate::ffi;
use crate::fix_point::FixT;

/// Copy `nitems` bytes from a C-allocated pointer into a `Vec<u8>`, then free the pointer.
///
/// # Safety
/// `ptr` must be a valid, non-null pointer to at least `nitems` bytes allocated by C `malloc`.
unsafe fn copy_and_free(ptr: *mut i8, nitems: c_int) -> Vec<u8> {
    assert!(!ptr.is_null(), "C compress function returned NULL");
    let len = nitems as usize;
    let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, len).to_vec() };
    unsafe { libc::free(ptr as *mut libc::c_void) };
    data
}

// ---------------------------------------------------------------------------
// C wrappers
// ---------------------------------------------------------------------------

fn c_compress_pos_int(
    pos: &[i32],
    natoms: i32,
    nframes: i32,
    prec_hi: u32,
    prec_lo: u32,
    speed: i32,
    algo: &mut [i32],
) -> Vec<u8> {
    let mut pos = pos.to_vec();
    let mut nitems: c_int = 0;
    unsafe {
        let ptr = ffi::tng_compress_pos_int(
            pos.as_mut_ptr(),
            natoms as c_int,
            nframes as c_int,
            prec_hi as c_ulong,
            prec_lo as c_ulong,
            speed as c_int,
            algo.as_mut_ptr(),
            &mut nitems,
        );
        copy_and_free(ptr, nitems)
    }
}

fn c_compress_pos(
    pos: &[f64],
    natoms: i32,
    nframes: i32,
    precision: f64,
    speed: i32,
    algo: &mut [i32],
) -> Vec<u8> {
    let mut pos = pos.to_vec();
    let mut nitems: c_int = 0;
    unsafe {
        let ptr = ffi::tng_compress_pos(
            pos.as_mut_ptr(),
            natoms as c_int,
            nframes as c_int,
            precision,
            speed as c_int,
            algo.as_mut_ptr(),
            &mut nitems,
        );
        copy_and_free(ptr, nitems)
    }
}

fn c_compress_pos_float(
    pos: &[f32],
    natoms: i32,
    nframes: i32,
    precision: f32,
    speed: i32,
    algo: &mut [i32],
) -> Vec<u8> {
    let mut pos = pos.to_vec();
    let mut nitems: c_int = 0;
    unsafe {
        let ptr = ffi::tng_compress_pos_float(
            pos.as_mut_ptr(),
            natoms as c_int,
            nframes as c_int,
            precision,
            speed as c_int,
            algo.as_mut_ptr(),
            &mut nitems,
        );
        copy_and_free(ptr, nitems)
    }
}

fn c_compress_vel_int(
    vel: &[i32],
    natoms: i32,
    nframes: i32,
    prec_hi: u32,
    prec_lo: u32,
    speed: i32,
    algo: &mut [i32],
) -> Vec<u8> {
    let mut vel = vel.to_vec();
    let mut nitems: c_int = 0;
    unsafe {
        let ptr = ffi::tng_compress_vel_int(
            vel.as_mut_ptr(),
            natoms as c_int,
            nframes as c_int,
            prec_hi as c_ulong,
            prec_lo as c_ulong,
            speed as c_int,
            algo.as_mut_ptr(),
            &mut nitems,
        );
        copy_and_free(ptr, nitems)
    }
}

fn c_compress_vel(
    vel: &[f64],
    natoms: i32,
    nframes: i32,
    precision: f64,
    speed: i32,
    algo: &mut [i32],
) -> Vec<u8> {
    let mut vel = vel.to_vec();
    let mut nitems: c_int = 0;
    unsafe {
        let ptr = ffi::tng_compress_vel(
            vel.as_mut_ptr(),
            natoms as c_int,
            nframes as c_int,
            precision,
            speed as c_int,
            algo.as_mut_ptr(),
            &mut nitems,
        );
        copy_and_free(ptr, nitems)
    }
}

fn c_compress_vel_float(
    vel: &[f32],
    natoms: i32,
    nframes: i32,
    precision: f32,
    speed: i32,
    algo: &mut [i32],
) -> Vec<u8> {
    let mut vel = vel.to_vec();
    let mut nitems: c_int = 0;
    unsafe {
        let ptr = ffi::tng_compress_vel_float(
            vel.as_mut_ptr(),
            natoms as c_int,
            nframes as c_int,
            precision,
            speed as c_int,
            algo.as_mut_ptr(),
            &mut nitems,
        );
        copy_and_free(ptr, nitems)
    }
}

fn c_compress_pos_find_algo(
    pos: &[f64],
    natoms: i32,
    nframes: i32,
    precision: f64,
    speed: i32,
    algo: &mut [i32],
) -> Vec<u8> {
    let mut pos = pos.to_vec();
    let mut nitems: c_int = 0;
    unsafe {
        let ptr = ffi::tng_compress_pos_find_algo(
            pos.as_mut_ptr(),
            natoms as c_int,
            nframes as c_int,
            precision,
            speed as c_int,
            algo.as_mut_ptr(),
            &mut nitems,
        );
        copy_and_free(ptr, nitems)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Generate position data for `natoms` atoms and `nframes` frames.
/// Values are small enough that quantization with precision 0.001 stays in range.
fn gen_pos_f64(natoms: usize, nframes: usize) -> Vec<f64> {
    let total = natoms * nframes * 3;
    (0..total).map(|i| (i as f64) * 0.01 + 0.1).collect()
}

fn gen_pos_f32(natoms: usize, nframes: usize) -> Vec<f32> {
    gen_pos_f64(natoms, nframes)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

#[test]
fn pos_int_matches_c() {
    // Use TRIPLET_INTER for remaining frames (compact output, avoids buffer overflow
    // that XTC3 causes with small inputs in both C and Rust).
    let quant = [-1i32, -2, -3, 10, 20, 30, 5, -10, 42, -7, 0, -99];
    let natoms = 2usize;
    let nframes = 2usize;
    let speed = 2usize;
    let prec_hi = FixT::from(123u32);
    let prec_lo = FixT::from(456u32);
    let algo_init = [
        compress::TNG_COMPRESS_ALGO_POS_XTC2,
        0,
        compress::TNG_COMPRESS_ALGO_POS_TRIPLET_INTER,
        0,
    ];

    // Rust
    let mut rust_quant = quant;
    let mut rust_algo = algo_init;
    let rust_result = compress::tng_compress_pos_int(
        &mut rust_quant,
        natoms as u32,
        nframes as u32,
        prec_hi,
        prec_lo,
        speed,
        &mut rust_algo,
    );

    // C
    let mut c_algo = algo_init;
    let c_result = c_compress_pos_int(
        &quant,
        natoms as i32,
        nframes as i32,
        u32::from(prec_hi),
        u32::from(prec_lo),
        speed as i32,
        &mut c_algo,
    );

    assert_eq!(c_result, rust_result, "pos_int: compressed data mismatch");
    assert_eq!(c_algo, rust_algo, "pos_int: algo mismatch");
}

#[test]
fn pos_double_matches_c() {
    let pos = gen_pos_f64(2, 2);
    let natoms = 2usize;
    let nframes = 2usize;
    let speed = 1usize;
    let precision = 0.001;
    let algo_init = [
        compress::TNG_COMPRESS_ALGO_POS_XTC2,
        0,
        compress::TNG_COMPRESS_ALGO_POS_TRIPLET_INTER,
        0,
    ];

    // Rust
    let mut rust_algo = algo_init;
    let rust_result =
        compress::tng_compress_pos(&pos, natoms, nframes, precision, speed, &mut rust_algo)
            .expect("Rust tng_compress_pos returned None");

    // C
    let mut c_algo = algo_init;
    let c_result = c_compress_pos(
        &pos,
        natoms as i32,
        nframes as i32,
        precision,
        speed as i32,
        &mut c_algo,
    );

    assert_eq!(
        c_result, rust_result,
        "pos_double: compressed data mismatch"
    );
    assert_eq!(c_algo, rust_algo, "pos_double: algo mismatch");
}

#[test]
fn pos_float_matches_c() {
    let pos = gen_pos_f32(2, 2);
    let natoms = 2usize;
    let nframes = 2usize;
    let speed = 1usize;
    let precision: f32 = 0.001;
    let algo_init = [
        compress::TNG_COMPRESS_ALGO_POS_XTC2,
        0,
        compress::TNG_COMPRESS_ALGO_POS_TRIPLET_INTER,
        0,
    ];

    // Rust
    let mut rust_algo = algo_init;
    let rust_result =
        compress::tng_compress_pos_float(&pos, natoms, nframes, precision, speed, &mut rust_algo)
            .expect("Rust tng_compress_pos_float returned None");

    // C
    let mut c_algo = algo_init;
    let c_result = c_compress_pos_float(
        &pos,
        natoms as i32,
        nframes as i32,
        precision,
        speed as i32,
        &mut c_algo,
    );

    assert_eq!(c_result, rust_result, "pos_float: compressed data mismatch");
    assert_eq!(c_algo, rust_algo, "pos_float: algo mismatch");
}

#[test]
fn vel_int_matches_c() {
    let vel = [100i32, -50, 25, -12, 6, -3, 42, -100, 200, -300, 400, -500];
    let natoms = 2usize;
    let nframes = 2usize;
    let speed = 2usize;
    let prec_hi = FixT::from(0u32);
    let prec_lo = FixT::from(1000u32);
    let algo_init = [
        compress::TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE,
        5,
        compress::TNG_COMPRESS_ALGO_VEL_TRIPLET_INTER,
        0,
    ];

    // Rust
    let mut rust_vel = vel;
    let mut rust_algo = algo_init;
    let rust_result = compress::tng_compress_vel_int(
        &mut rust_vel,
        natoms,
        nframes,
        prec_hi,
        prec_lo,
        speed,
        &mut rust_algo,
    );

    // C
    let mut c_algo = algo_init;
    let c_result = c_compress_vel_int(
        &vel,
        natoms as i32,
        nframes as i32,
        u32::from(prec_hi),
        u32::from(prec_lo),
        speed as i32,
        &mut c_algo,
    );

    assert_eq!(c_result, rust_result, "vel_int: compressed data mismatch");
    assert_eq!(c_algo, rust_algo, "vel_int: algo mismatch");
}

#[test]
fn vel_double_matches_c() {
    let vel = [
        0.5, -0.3, 0.1, 0.2, -0.4, 0.6, 0.8, -0.1, 0.3, 0.7, -0.5, 0.9,
    ];
    let natoms = 2usize;
    let nframes = 2usize;
    let speed = 1usize;
    let precision = 0.001;
    let algo_init = [
        compress::TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE,
        5,
        compress::TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER,
        5,
    ];

    // Rust
    let mut rust_algo = algo_init;
    let rust_result =
        compress::tng_compress_vel(&vel, natoms, nframes, precision, speed, &mut rust_algo)
            .expect("Rust tng_compress_vel returned None");

    // C
    let mut c_algo = algo_init;
    let c_result = c_compress_vel(
        &vel,
        natoms as i32,
        nframes as i32,
        precision,
        speed as i32,
        &mut c_algo,
    );

    assert_eq!(
        c_result, rust_result,
        "vel_double: compressed data mismatch"
    );
    assert_eq!(c_algo, rust_algo, "vel_double: algo mismatch");
}

#[test]
fn vel_float_matches_c() {
    let vel: Vec<f32> = [
        0.5f32, -0.3, 0.1, 0.2, -0.4, 0.6, 0.8, -0.1, 0.3, 0.7, -0.5, 0.9,
    ]
    .into();
    let natoms = 2usize;
    let nframes = 2usize;
    let speed = 1usize;
    let precision: f32 = 0.001;
    let algo_init = [
        compress::TNG_COMPRESS_ALGO_VEL_STOPBIT_ONETOONE,
        5,
        compress::TNG_COMPRESS_ALGO_VEL_STOPBIT_INTER,
        5,
    ];

    // Rust
    let mut rust_algo = algo_init;
    let rust_result =
        compress::tng_compress_vel_float(&vel, natoms, nframes, precision, speed, &mut rust_algo)
            .expect("Rust tng_compress_vel_float returned None");

    // C
    let mut c_algo = algo_init;
    let c_result = c_compress_vel_float(
        &vel,
        natoms as i32,
        nframes as i32,
        precision,
        speed as i32,
        &mut c_algo,
    );

    assert_eq!(c_result, rust_result, "vel_float: compressed data mismatch");
    assert_eq!(c_algo, rust_algo, "vel_float: algo mismatch");
}

#[test]
fn find_algo_pos_matches_c() {
    // Use enough atoms that the buffer (n*nf*14 + 44) is large enough for any algo.
    let natoms = 20usize;
    let nframes = 2usize;
    let pos = gen_pos_f64(natoms, nframes);
    let speed = 1usize; // fast algos only — avoids BWLZH which may overflow small buffers
    let precision = 0.001;

    // Rust: algo = [-1,-1,-1,-1] triggers find-algo behavior
    let mut rust_algo = [-1i32, -1, -1, -1];
    let rust_result =
        compress::tng_compress_pos(&pos, natoms, nframes, precision, speed, &mut rust_algo)
            .expect("Rust tng_compress_pos returned None");

    // C: use the explicit find_algo function
    let mut c_algo = [0i32; 4];
    let c_result = c_compress_pos_find_algo(
        &pos,
        natoms as i32,
        nframes as i32,
        precision,
        speed as i32,
        &mut c_algo,
    );

    assert_eq!(c_algo, rust_algo, "find_algo_pos: algo mismatch");
    assert_eq!(
        c_result, rust_result,
        "find_algo_pos: compressed data mismatch"
    );
}
