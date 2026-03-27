use super::c_helpers::*;
use crate::compress;
use crate::fix_point::FixT;

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
        compress::tng_compress_pos(&pos, natoms, nframes, precision, speed, &mut rust_algo)
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
        compress::tng_compress_vel(&vel, natoms, nframes, precision, speed, &mut rust_algo)
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
