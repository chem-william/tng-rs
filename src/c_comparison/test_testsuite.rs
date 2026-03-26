/// Ports of the C compression testsuite tests (vendor/tng/src/tests/compression/).
///
/// Each C test is defined by a testN.h header with #define parameters.
/// The test mirrors the C structure: a GEN phase compresses data into an
/// in-memory file, then a read phase decompresses each chunk with C's
/// `tng_compress_uncompress` and checks precision against re-generated data.
use super::testsuite_data::{TestParams, genibox, realbox};
use crate::compress;
use crate::trajectory::compress_uncompress;

const FUDGE: f64 = 1.1; // 10% off target precision is acceptable.

// ---------------------------------------------------------------------------
// File format: [natoms:u32le] then per chunk: [nframes:u32le] [nitems:u32le] [blob]
// ---------------------------------------------------------------------------

fn write_int_le(out: &mut Vec<u8>, val: i32) {
    out.extend_from_slice(&(val as u32).to_le_bytes());
}

fn read_int_le(data: &[u8], pos: &mut usize) -> Option<i32> {
    if *pos + 4 > data.len() {
        return None;
    }
    let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as i32;
    *pos += 4;
    Some(val)
}

// ---------------------------------------------------------------------------
// Write side (mirrors GEN path)
// ---------------------------------------------------------------------------

/// `struct tng_file` from testsuite.c (write mode).
struct TngFileWrite {
    natoms: usize,
    chunky: usize,
    precision: f64,
    speed: usize,
    initial_coding: i32,
    initial_coding_parameter: i32,
    coding: i32,
    coding_parameter: i32,
    nframes: usize,
    pos: Vec<f64>,
    file: Vec<u8>,
}

/// `open_tng_file_write` from testsuite.c.
fn open_tng_file_write(params: &TestParams) -> TngFileWrite {
    let mut file = Vec::new();
    write_int_le(&mut file, params.natoms as i32);
    TngFileWrite {
        natoms: params.natoms,
        chunky: params.chunky,
        precision: params.precision,
        speed: params.speed,
        initial_coding: params.initial_coding,
        initial_coding_parameter: params.initial_coding_parameter,
        coding: params.coding,
        coding_parameter: params.coding_parameter,
        nframes: 0,
        pos: vec![0.0f64; params.natoms * params.chunky * 3],
        file,
    }
}

/// `flush_tng_frames` from testsuite.c — compresses the accumulated chunk
/// and appends it to the in-memory file.
fn flush_tng_frames(tng_file: &mut TngFileWrite) {
    let n = tng_file.natoms;
    let nframes = tng_file.nframes;
    let pos = &tng_file.pos[..n * nframes * 3];

    let mut algo = [
        tng_file.initial_coding,
        tng_file.initial_coding_parameter,
        tng_file.coding,
        tng_file.coding_parameter,
    ];

    let buf = compress::tng_compress_pos(
        pos,
        n,
        nframes,
        tng_file.precision,
        tng_file.speed,
        &mut algo,
    )
    .expect("Rust compression returned None");

    write_int_le(&mut tng_file.file, nframes as i32);
    write_int_le(&mut tng_file.file, buf.len() as i32);
    tng_file.file.extend_from_slice(&buf);

    tng_file.initial_coding = algo[0];
    tng_file.initial_coding_parameter = algo[1];
    tng_file.coding = algo[2];
    tng_file.coding_parameter = algo[3];
    tng_file.nframes = 0;
}

/// `write_tng_file` from testsuite.c — appends a frame, flushing when full.
fn write_tng_file(tng_file: &mut TngFileWrite, pos: &[f64]) {
    let n = tng_file.natoms;
    let offset = tng_file.nframes * n * 3;
    tng_file.pos[offset..offset + n * 3].copy_from_slice(&pos[..n * 3]);
    tng_file.nframes += 1;
    if tng_file.nframes == tng_file.chunky {
        flush_tng_frames(tng_file);
    }
}

/// `close_tng_file_write` from testsuite.c — flushes remaining frames.
fn close_tng_file_write(tng_file: &mut TngFileWrite) {
    if tng_file.nframes > 0 {
        flush_tng_frames(tng_file);
    }
}

// ---------------------------------------------------------------------------
// Read side (mirrors read path)
// ---------------------------------------------------------------------------

/// `struct tng_file` from testsuite.c (read mode).
struct TngFileRead<'a> {
    data: &'a [u8],
    cursor: usize,
    natoms: usize,
    nframes: usize,
    nframes_delivered: usize,
    pos: Vec<f64>,
}

/// `open_tng_file_read` from testsuite.c.
fn open_tng_file_read(data: &[u8]) -> TngFileRead<'_> {
    let mut cursor = 0;
    let natoms = read_int_le(data, &mut cursor).expect("failed to read natoms") as usize;
    TngFileRead {
        data,
        cursor,
        natoms,
        nframes: 0,
        nframes_delivered: 0,
        pos: Vec::new(),
    }
}

/// `read_tng_file` from testsuite.c — returns one frame of decompressed data.
/// Uses Rust's `compress_uncompress` to decompress.
fn read_tng_file(tng_file: &mut TngFileRead, pos: &mut [f64]) -> Result<(), ()> {
    if tng_file.nframes == tng_file.nframes_delivered {
        let nframes = read_int_le(tng_file.data, &mut tng_file.cursor).ok_or(())? as usize;
        let nitems = read_int_le(tng_file.data, &mut tng_file.cursor).ok_or(())? as usize;
        let blob = &tng_file.data[tng_file.cursor..tng_file.cursor + nitems];
        tng_file.cursor += nitems;

        tng_file.pos.resize(tng_file.natoms * nframes * 3, 0.0);
        compress_uncompress(blob, &mut tng_file.pos).expect("Rust decompress failed");
        tng_file.nframes = nframes;
        tng_file.nframes_delivered = 0;
    }

    let n = tng_file.natoms;
    let offset = tng_file.nframes_delivered * n * 3;
    pos[..n * 3].copy_from_slice(&tng_file.pos[offset..offset + n * 3]);
    tng_file.nframes_delivered += 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// equalarr (testsuite.c lines 230-256)
// ---------------------------------------------------------------------------

fn equalarr(arr1: &[f64], arr2: &[f64], prec: f64, natoms: usize) -> f64 {
    let mut maxdiff: f64 = 0.0;
    for i in 0..natoms {
        for j in 0..3 {
            let diff = (arr1[i * 3 + j] - arr2[i * 3 + j]).abs();
            if diff > maxdiff {
                maxdiff = diff;
            }
        }
    }
    assert!(
        maxdiff <= prec * 0.5 * FUDGE,
        "precision exceeded: max_diff={maxdiff}, tolerance={}",
        prec * 0.5 * FUDGE,
    );
    maxdiff
}

// ---------------------------------------------------------------------------
// algotest (testsuite.c lines 717-923)
// ---------------------------------------------------------------------------

/// Mirrors `algotest()` from testsuite.c.
/// GEN phase: generate data, compress with Rust, write to in-memory file.
/// Read phase: re-generate data, decompress with Rust, check precision.
fn algotest(params: &TestParams) {
    let natoms = params.natoms;
    let mut intbox = vec![0i32; natoms * 3];
    let mut box1 = vec![0.0f64; natoms * 3];

    // --- GEN phase ---
    let mut dumpfile = open_tng_file_write(params);

    for iframe in 0..params.nframes {
        genibox(&mut intbox, iframe as i32, params);
        realbox(
            &intbox,
            &mut box1,
            3,
            natoms,
            params.genprecision,
            params.scale,
        );
        write_tng_file(&mut dumpfile, &box1);
    }

    close_tng_file_write(&mut dumpfile);

    // GEN filesize check
    let filesize = dumpfile.file.len() as f64;
    if filesize > 0.0 {
        let diff = (filesize - params.expected_filesize).abs() / params.expected_filesize;
        assert!(
            diff <= 0.05,
            "filesize {filesize} too far from expected {} ({:.1}%)",
            params.expected_filesize,
            diff * 100.0,
        );
    }

    // --- Read phase ---
    let mut reader = open_tng_file_read(&dumpfile.file);
    let mut box2 = vec![0.0f64; natoms * 3];

    for iframe in 0..params.nframes {
        genibox(&mut intbox, iframe as i32, params);
        realbox(
            &intbox,
            &mut box1,
            3,
            natoms,
            params.genprecision,
            params.scale,
        );
        read_tng_file(&mut reader, &mut box2).expect("read error");
        equalarr(&box1, &box2, params.precision, natoms);
    }
}

// ---------------------------------------------------------------------------
// Test parameters (from vendor/tng/src/tests/compression/testN.h)
// ---------------------------------------------------------------------------

// Initial coding. Intra frame triple algorithm. Cubic cell
fn test1_params() -> TestParams {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3, // TRIPLET_INTRA
        initial_coding_parameter: -1,
        coding: 1, // TRIPLET_INTER
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        stride: 3,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2776230.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. XTC2 algorithm. Cubic cell
fn test2_params() -> TestParams {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        stride: 3,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2796171.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. Triplet one-to-one algorithm . Cubic cell
fn test3_params() -> TestParams {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 7,
        initial_coding_parameter: -1,
        coding: 1,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        stride: 3,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 4356773.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. BWLZH intra algorithm. Cubic cell
fn test4_params() -> TestParams {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 9,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        stride: 3,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2572043.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. XTC3 algorithm. Cubic cell
fn test5_params() -> TestParams {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 10,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        stride: 3,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 3346179.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. XTC2 algorithm. Cubic cell
fn test6_params() -> TestParams {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        stride: 3,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2736662.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. Stopbit interframe algorithm. Cubic cell
fn test7_params() -> TestParams {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: -1,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        stride: 3,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2545049.0,
        regular: false,
        velintmul: None,
    }
}

#[test]
fn test1() {
    algotest(&test1_params());
}

#[test]
fn test2() {
    algotest(&test2_params());
}

#[test]
fn test3() {
    algotest(&test3_params());
}

#[test]
fn test4() {
    algotest(&test4_params());
}

#[test]
fn test5() {
    algotest(&test5_params());
}

#[test]
fn test6() {
    algotest(&test6_params());
}

#[test]
fn test7() {
    algotest(&test7_params());
}
