/// Ports of the C compression testsuite tests (vendor/tng/src/tests/compression/).
///
/// Each C test is defined by a testN.h header with #define parameters.
/// The test mirrors the C structure: a GEN phase compresses data into an
/// in-memory file, then a read phase decompresses each chunk with C's
/// `tng_compress_uncompress` and checks precision against re-generated data.
use super::testsuite_data::{TestParams, genibox, genivelbox, realbox, realvelbox};
use crate::compress::tng_compress_vel;
use crate::compress::{Float, tng_compress_pos};
use crate::trajectory::compress_uncompress;

const FUDGE: f64 = 1.1; // 10% off target precision is acceptable.

// ---------------------------------------------------------------------------
// File format: [natoms:u32le] then per chunk:
//   [nframes:u32le] [pos_nitems:u32le] [pos_blob]
//   (if writevel) [vel_nitems:u32le] [vel_blob]
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
struct TngFileWrite<T: Float> {
    natoms: usize,
    chunky: usize,
    precision: T,
    velprecision: T,
    speed: usize,
    initial_coding: i32,
    initial_coding_parameter: i32,
    coding: i32,
    coding_parameter: i32,
    initial_velcoding: i32,
    initial_velcoding_parameter: i32,
    velcoding: i32,
    velcoding_parameter: i32,
    nframes: usize,
    pos: Vec<T>,
    vel: Vec<T>,
    file: Vec<u8>,
}

/// `open_tng_file_write` from testsuite.c.
fn open_tng_file_write<T: Float>(params: &TestParams<T>) -> TngFileWrite<T> {
    let mut file = Vec::new();
    write_int_le(&mut file, params.natoms as i32);
    let vel = if params.writevel {
        vec![T::from_f64(0.0); params.natoms * params.chunky * 3]
    } else {
        Vec::new()
    };
    TngFileWrite {
        natoms: params.natoms,
        chunky: params.chunky,
        precision: params.precision,
        velprecision: params.velprecision,
        initial_coding: params.initial_coding,
        initial_coding_parameter: params.initial_coding_parameter,
        coding: params.coding,
        coding_parameter: params.coding_parameter,
        initial_velcoding: params.initial_velcoding,
        initial_velcoding_parameter: params.initial_velcoding_parameter,
        velcoding: params.velcoding,
        velcoding_parameter: params.velcoding_parameter,
        speed: params.speed,
        nframes: 0,
        pos: vec![T::from_f64(0.0); params.natoms * params.chunky * 3],
        vel,
        file,
    }
}

/// `flush_tng_frames` from testsuite.c — compresses the accumulated chunk
/// and appends it to the in-memory file.
fn flush_tng_frames<T: Float>(tng_file: &mut TngFileWrite<T>, writevel: bool) {
    let n = tng_file.natoms;
    let nframes = tng_file.nframes;
    let pos = &tng_file.pos[..n * nframes * 3];

    write_int_le(&mut tng_file.file, nframes as i32);
    let mut algo = [
        tng_file.initial_coding,
        tng_file.initial_coding_parameter,
        tng_file.coding,
        tng_file.coding_parameter,
    ];

    let buf = tng_compress_pos(
        pos,
        n,
        nframes,
        tng_file.precision,
        tng_file.speed,
        &mut algo,
    )
    .expect("pos compression returned None");
    tng_file.initial_coding = algo[0];
    tng_file.initial_coding_parameter = algo[1];
    tng_file.coding = algo[2];
    tng_file.coding_parameter = algo[3];
    write_int_le(&mut tng_file.file, buf.len() as i32);
    tng_file.file.extend_from_slice(&buf);

    if writevel {
        let vel = &tng_file.vel[..n * nframes * 3];
        let mut algo = [
            tng_file.initial_velcoding,
            tng_file.initial_velcoding_parameter,
            tng_file.velcoding,
            tng_file.velcoding_parameter,
        ];

        let buf = tng_compress_vel(
            vel,
            n,
            nframes,
            tng_file.velprecision,
            tng_file.speed,
            &mut algo,
        )
        .expect("vel compression returned None");
        tng_file.initial_velcoding = algo[0];
        tng_file.initial_velcoding_parameter = algo[1];
        tng_file.velcoding = algo[2];
        tng_file.velcoding_parameter = algo[3];
        write_int_le(&mut tng_file.file, buf.len() as i32);
        tng_file.file.extend_from_slice(&buf);
    }

    tng_file.nframes = 0;
}

/// `write_tng_file` from testsuite.c — appends a frame, flushing when full.
fn write_tng_file<T: Float>(tng_file: &mut TngFileWrite<T>, pos: &[T], vel: &[T], writevel: bool) {
    let n = tng_file.natoms;
    let offset = tng_file.nframes * n * 3;
    tng_file.pos[offset..offset + n * 3].copy_from_slice(&pos[..n * 3]);
    if writevel {
        tng_file.vel[offset..offset + n * 3].copy_from_slice(&vel[..n * 3]);
    }
    tng_file.nframes += 1;
    if tng_file.nframes == tng_file.chunky {
        flush_tng_frames(tng_file, writevel);
    }
}

/// `close_tng_file_write` from testsuite.c — flushes remaining frames.
fn close_tng_file_write<T: Float>(tng_file: &mut TngFileWrite<T>, writevel: bool) {
    if tng_file.nframes > 0 {
        flush_tng_frames(tng_file, writevel);
    }
}

// ---------------------------------------------------------------------------
// Read side (mirrors read path)
// ---------------------------------------------------------------------------

/// `struct tng_file` from testsuite.c (read mode).
struct TngFileRead<'a, T: Float> {
    data: &'a [u8],
    cursor: usize,
    natoms: usize,
    nframes: usize,
    nframes_delivered: usize,
    pos: Vec<T>,
    vel: Vec<T>,
}

/// `open_tng_file_read` from testsuite.c.
fn open_tng_file_read<T: Float>(data: &[u8]) -> TngFileRead<'_, T> {
    let mut cursor = 0;
    let natoms = read_int_le(data, &mut cursor).expect("failed to read natoms") as usize;
    TngFileRead {
        data,
        cursor,
        natoms,
        nframes: 0,
        nframes_delivered: 0,
        pos: Vec::new(),
        vel: Vec::new(),
    }
}

/// `read_tng_file` from testsuite.c — returns one frame of decompressed data.
fn read_tng_file<T: Float>(
    tng_file: &mut TngFileRead<T>,
    pos: &mut [T],
    vel: &mut [T],
    writevel: bool,
) -> Result<(), ()> {
    if tng_file.nframes == tng_file.nframes_delivered {
        let nframes = read_int_le(tng_file.data, &mut tng_file.cursor).ok_or(())? as usize;
        let nitems = read_int_le(tng_file.data, &mut tng_file.cursor).ok_or(())? as usize;
        let blob = &tng_file.data[tng_file.cursor..tng_file.cursor + nitems];
        tng_file.cursor += nitems;

        tng_file
            .pos
            .resize(tng_file.natoms * nframes * 3, T::from_f64(0.0));
        if writevel {
            tng_file
                .vel
                .resize(tng_file.natoms * nframes * 3, T::from_f64(0.0));
        }
        compress_uncompress(blob, &mut tng_file.pos).expect("decompress failed");

        if writevel {
            let nitems = read_int_le(tng_file.data, &mut tng_file.cursor).ok_or(())? as usize;
            let blob = &tng_file.data[tng_file.cursor..tng_file.cursor + nitems];
            tng_file.cursor += nitems;
            compress_uncompress(blob, &mut tng_file.vel).expect("vel decompress failed");
        }
        tng_file.nframes = nframes;
        tng_file.nframes_delivered = 0;
    }

    let n = tng_file.natoms;
    let offset = tng_file.nframes_delivered * n * 3;
    pos[..n * 3].copy_from_slice(&tng_file.pos[offset..offset + n * 3]);
    if writevel {
        vel[..n * 3].copy_from_slice(&tng_file.vel[offset..offset + n * 3]);
    }
    tng_file.nframes_delivered += 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// equalarr (testsuite.c lines 230-256)
// ---------------------------------------------------------------------------

fn equalarr<W: Float, R: Float>(arr1: &[W], arr2: &[R], prec: R, natoms: usize) -> f64 {
    let mut maxdiff = R::from_f64(0.0);
    for i in 0..natoms {
        for j in 0..3 {
            let diff = R::from_f64((W::to_f64(arr1[i * 3 + j]) - R::to_f64(arr2[i * 3 + j])).abs());
            if diff > maxdiff {
                maxdiff = diff;
            }
        }
    }
    assert!(
        maxdiff <= prec * R::from_f64(0.5 * FUDGE),
        "precision exceeded: max_diff={maxdiff}, tolerance={}",
        prec * R::from_f64(0.5 * FUDGE),
    );
    R::to_f64(maxdiff)
}

// ---------------------------------------------------------------------------
// algotest (testsuite.c lines 717-923)
// ---------------------------------------------------------------------------

/// Mirrors `algotest()` from testsuite.c.
/// GEN phase: generate data, compress, write to in-memory file.
/// Read phase: re-generate data, decompress, check precision.
fn algotest<W: Float, R: Float>(params: &TestParams<W>) {
    let natoms = params.natoms;
    let mut intbox = vec![0i32; natoms * 3];
    let mut intvelbox = vec![0i32; natoms * 3];
    let mut box1 = vec![W::from_f64(0.0); natoms * 3];
    let mut velbox1 = vec![W::from_f64(0.0); natoms * 3];

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
        if params.writevel {
            genivelbox(&mut intvelbox, iframe as i32, params);
            realvelbox(
                &intvelbox,
                &mut velbox1,
                3,
                natoms,
                params.genvelprecision,
                params.scale,
            );
        }
        write_tng_file(&mut dumpfile, &box1, &velbox1, params.writevel);
    }

    close_tng_file_write(&mut dumpfile, params.writevel);

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
    // we regenerate box1 and velbox1 to make sure they are in the R(eader) type
    let mut box1 = vec![R::from_f64(0.0); natoms * 3];
    let mut velbox1 = vec![R::from_f64(0.0); natoms * 3];
    let mut box2 = vec![R::from_f64(0.0); natoms * 3];
    let mut velbox2 = vec![R::from_f64(0.0); natoms * 3];

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
        if params.writevel {
            genivelbox(&mut intvelbox, iframe as i32, params);
            realvelbox(
                &intvelbox,
                &mut velbox1,
                3,
                natoms,
                params.genvelprecision,
                params.scale,
            );
        }
        read_tng_file(&mut reader, &mut box2, &mut velbox2, params.writevel).expect("read error");
        equalarr(
            &box1,
            &box2,
            R::from_f64(W::to_f64(params.precision)),
            natoms,
        );
        if params.writevel {
            equalarr(
                &velbox1,
                &velbox2,
                R::from_f64(W::to_f64(params.velprecision)),
                natoms,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test parameters (from vendor/tng/src/tests/compression/testN.h)
// ---------------------------------------------------------------------------

// Initial coding. Intra frame triple algorithm. Cubic cell
fn test1_params() -> TestParams<f64> {
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2776230.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. XTC2 algorithm. Cubic cell
fn test2_params() -> TestParams<f64> {
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2796171.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. Triplet one-to-one algorithm . Cubic cell
fn test3_params() -> TestParams<f64> {
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 4356773.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. BWLZH intra algorithm. Cubic cell
fn test4_params() -> TestParams<f64> {
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2572043.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. XTC3 algorithm. Cubic cell
fn test5_params() -> TestParams<f64> {
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 3346179.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. XTC2 algorithm. Cubic cell
fn test6_params() -> TestParams<f64> {
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2736662.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. Stopbit interframe algorithm. Cubic cell
fn test7_params() -> TestParams<f64> {
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2545049.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. Stopbit interframe algorithm with intraframe compression as initial. Cubic cell
fn test8_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2544876.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. Triple interframe algorithm. Cubic cell
fn test9_params() -> TestParams<f64> {
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
        coding: 2,
        coding_parameter: -1,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2418212.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. Triple intraframe algorithm. Cubic cell
fn test10_params() -> TestParams<f64> {
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
        coding: 3,
        coding_parameter: -1,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2728492.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. Triple one-to-one algorithm. Cubic cell
fn test11_params() -> TestParams<f64> {
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
        coding: 7,
        coding_parameter: -1,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 4293415.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. BWLZH interframe algorithm. Cubic cell
fn test12_params() -> TestParams<f64> {
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
        coding: 8,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 894421.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. BWLZH intraframe algorithm. Cubic cell
fn test13_params() -> TestParams<f64> {
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
        coding: 9,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 840246.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. XTC3 algorithm. Cubic cell
fn test14_params() -> TestParams<f64> {
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
        coding: 10,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 1401016.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. Automatic selection of algorithms. Cubic cell
fn test15_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: -1,
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
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2776230.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. Automatic selection of algorithms. Cubic cell
fn test16_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: -1,
        initial_coding_parameter: -1,
        coding: -1,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 6,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 838168.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding of velocities. Stopbits one-to-one. Cubic cell
fn test17_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 7336171.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding of velocities. Triplet one-to-one. Cubic cell
fn test18_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: 3,
        initial_velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 7089695.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding of velocities. BWLZH one-to-one. Cubic cell
fn test19_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 50,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: 0,
        velcoding: 0,
        velcoding_parameter: 0,
        initial_velcoding: 9,
        initial_velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 208809.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. Intra frame triple algorithm. High accuracy. Cubic cell.
fn test41_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 1,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [1610612736, 1610612736, 1610612736],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 53179342.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. XTC2 algorithm. High accuracy. Cubic cell.
fn test42_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 1,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [1610612736, 1610612736, 1610612736],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 57283715.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. XTC3 algorithm. High accuracy. Cubic cell.
fn test43_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 1,
        nframes: 10,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 10,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [1610612736, 1610612736, 1610612736],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 3783912.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. Intra frame BWLZH algorithm. High accuracy. Cubic cell.
fn test44_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 1,
        nframes: 10,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 9,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [1610612736, 1610612736, 1610612736],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 1436901.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Stop bits algorithm. High accuracy. Cubic cell.
fn test45_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [1610612736, 1610612736, 1610612736],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 36794379.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Inter frame triple algorithm. High accuracy. Cubic cell.
fn test46_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 2,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [1610612736, 1610612736, 1610612736],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 34508770.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding.  Intra frame triple algorithm. High accuracy. Cubic cell.
fn test47_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 3,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [1610612736, 1610612736, 1610612736],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 53174711.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. XTC2 algorithm. High accuracy. Cubic cell.
fn test48_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 55638414.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. XTC3 algorithm. High accuracy. Cubic cell.
fn test49_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 20,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 10,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 3585605.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Intra frame BWLZH algorithm. High accuracy. Cubic cell.
fn test50_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 20,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 9,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 3143379.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Inter frame BWLZH algorithm. High accuracy. Cubic cell.
fn test51_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 20,
        scale: 0.5,
        precision: 1e-8,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 8,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: 0,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 0.1,
        expected_filesize: 2897696.0,
        regular: false,
        velintmul: None,
    }
}

// Velocity coding. Stop bits algorithm. High accuracy. Cubic cell.
fn test52_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: true,
        velprecision: 1e-8,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        velcoding: 1,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 1e-8,
        expected_filesize: 173083705.0,
        regular: false,
        velintmul: Some(100000),
    }
}

// Velocity coding. Triple algorithm. High accuracy. Cubic cell.
fn test53_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: true,
        velprecision: 1e-8,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        velcoding: 3,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 1e-8,
        expected_filesize: 168548573.0,
        regular: false,
        velintmul: Some(100000),
    }
}

// Velocity coding. Interframe triple algorithm. High accuracy. Cubic cell.
fn test54_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: true,
        velprecision: 1e-8,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        velcoding: 2,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 1e-8,
        expected_filesize: 161798573.0,
        regular: false,
        velintmul: Some(100000),
    }
}

// Velocity coding. Interframe stop-bits algorithm. High accuracy. Cubic cell.
fn test55_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 100,
        scale: 0.5,
        precision: 1e-8,
        writevel: true,
        velprecision: 1e-8,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        velcoding: 6,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 1e-8,
        expected_filesize: 166298533.0,
        regular: false,
        velintmul: Some(100000),
    }
}

// Velocity coding. Intraframe BWLZH algorithm. High accuracy. Cubic cell.
fn test56_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 20,
        scale: 0.5,
        precision: 1e-8,
        writevel: true,
        velprecision: 1e-8,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        velcoding: 9,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 1e-8,
        expected_filesize: 23390767.0,
        regular: false,
        velintmul: Some(100000),
    }
}

// Velocity coding. Interframe BWLZH algorithm. High accuracy. Cubic cell.
fn test57_params() -> TestParams<f64> {
    TestParams {
        natoms: 100000,
        chunky: 10,
        nframes: 20,
        scale: 0.5,
        precision: 1e-8,
        writevel: true,
        velprecision: 1e-8,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: -1,
        initial_velcoding_parameter: -1,
        velcoding: 8,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [805306368, 805306368, 805306368],
        speed: 5,
        framescale: 1,
        genprecision: 1e-8,
        genvelprecision: 1e-8,
        expected_filesize: 13817974.0,
        regular: false,
        velintmul: Some(100000),
    }
}

// Coding. Test float
fn test58_params() -> TestParams<f32> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 3,
        initial_velcoding_parameter: -1,
        velcoding: 3,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 6986313.0,
        regular: false,
        velintmul: None,
    }
}

// Coding. Test write float, read double
fn test59_params() -> TestParams<f32> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 3,
        initial_velcoding_parameter: -1,
        velcoding: 3,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 6986313.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Inter frame BWLZH algorithm. Large system. Cubic cell.
fn test40_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 2,
        nframes: 4,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 8,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 63822378.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Intra frame BWLZH algorithm. Large system. Cubic cell.
fn test39_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 2,
        nframes: 4,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 9,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 67631371.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. XTC3 algorithm. Large system. Cubic cell.
fn test38_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 2,
        nframes: 4,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 10,
        initial_coding_parameter: 0,
        coding: 10,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 63482016.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. XTC2 algorithm. Large system. Cubic cell.
fn test37_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 2,
        nframes: 10,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 301463256.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Intra frame triple algorithm. Large system. Cubic cell.
fn test36_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 2,
        nframes: 10,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 3,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 290800607.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Inter frame triple algorithm. Large system. Cubic cell.
fn test35_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 2,
        nframes: 10,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 2,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 243598962.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Stop bits algorithm. Large system. Cubic cell.
fn test34_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 2,
        nframes: 10,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 250247372.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. Intra frame BWLZH algorithm. Large system. Cubic cell.
fn test33_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 1,
        nframes: 2,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 9,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 7121047.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. XTC3 algorithm. Large system. Cubic cell.
fn test32_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 1,
        nframes: 2,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 10,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 31668187.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. XTC2 algorithm. Large system. Cubic cell.
fn test31_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 1,
        nframes: 10,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 301463456.0,
        regular: false,
        velintmul: None,
    }
}

// Initial coding. Intra frame triple algorithm. Large system. Cubic cell.
fn test30_params() -> TestParams<f64> {
    TestParams {
        natoms: 5000000,
        chunky: 1,
        nframes: 10,
        scale: 1.0,
        precision: 1.0,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 3,
        initial_coding_parameter: -1,
        coding: 1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 4,
        velcoding_parameter: 0,
        intmin: [-536870911, -536870911, -536870911],
        intmax: [536870911, 536870911, 536870911],
        speed: 5,
        framescale: 1,
        genprecision: 1.0,
        genvelprecision: 0.1,
        expected_filesize: 280198420.0,
        regular: false,
        velintmul: None,
    }
}

// Position coding. Autoselect algorithm. Repetitive molecule. Cubic cell.
fn test29_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: -1,
        initial_coding_parameter: -1,
        coding: -1,
        coding_parameter: -1,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 0,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 228148.0,
        regular: true,
        velintmul: None,
    }
}

// Initial coding. Autoselect algorithm. Repetitive molecule. Cubic cell.
fn test28_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 1,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: -1,
        initial_coding_parameter: -1,
        coding: 0,
        coding_parameter: 0,
        initial_velcoding: 0,
        initial_velcoding_parameter: -1,
        velcoding: 0,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 1677619.0,
        regular: true,
        velintmul: None,
    }
}

// XTC3 algorithm. Orthorhombic cell.
fn test27_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 200,
        scale: 0.1,
        precision: 0.01,
        writevel: false,
        velprecision: 0.1,
        initial_coding: 10,
        initial_coding_parameter: 0,
        coding: 10,
        coding_parameter: 0,
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        velcoding: 9,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [20000, 10000, 30000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 282600.0,
        regular: false,
        velintmul: None,
    }
}

// XTC2 algorithm. Orthorhombic cell.
fn test26_params() -> TestParams<f64> {
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
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        velcoding: 9,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [20000, 10000, 30000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 2861948.0,
        regular: false,
        velintmul: None,
    }
}

// Coding of velocities. BWLZH one-to-one. Cubic cell.
fn test25_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 25,
        nframes: 50,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        velcoding: 9,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 154753.0,
        regular: false,
        velintmul: None,
    }
}

// Coding of velocities. BWLZH interframe. Cubic cell.
fn test24_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 25,
        nframes: 50,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        velcoding: 8,
        velcoding_parameter: 0,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 153520.0,
        regular: false,
        velintmul: None,
    }
}

// Coding of velocities. Stopbit interframe. Cubic cell.
fn test23_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        velcoding: 6,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 6494602.0,
        regular: false,
        velintmul: None,
    }
}

// Coding of velocities. Triplet one-to-one. Cubic cell.
fn test22_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        velcoding: 3,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 6988699.0,
        regular: false,
        velintmul: None,
    }
}

// Coding of velocities. Triplet inter. Cubic cell.
fn test21_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        velcoding: 2,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 6214307.0,
        regular: false,
        velintmul: None,
    }
}

// Coding of velocities. Stopbit one-to-one. Cubic cell.
fn test20_params() -> TestParams<f64> {
    TestParams {
        natoms: 1000,
        chunky: 100,
        nframes: 1000,
        scale: 0.1,
        precision: 0.01,
        writevel: true,
        velprecision: 0.1,
        initial_coding: 5,
        initial_coding_parameter: 0,
        coding: 5,
        coding_parameter: 0,
        initial_velcoding: 1,
        initial_velcoding_parameter: -1,
        velcoding: 1,
        velcoding_parameter: -1,
        intmin: [0, 0, 0],
        intmax: [10000, 10000, 10000],
        speed: 5,
        framescale: 1,
        genprecision: 0.01,
        genvelprecision: 0.1,
        expected_filesize: 7237102.0,
        regular: false,
        velintmul: None,
    }
}

#[test]
fn test1() {
    algotest::<f64, f64>(&test1_params());
}

#[test]
fn test2() {
    algotest::<f64, f64>(&test2_params());
}

#[test]
fn test3() {
    algotest::<f64, f64>(&test3_params());
}

#[test]
fn test4() {
    algotest::<f64, f64>(&test4_params());
}

#[test]
fn test5() {
    algotest::<f64, f64>(&test5_params());
}

#[test]
fn test6() {
    algotest::<f64, f64>(&test6_params());
}

#[test]
fn test7() {
    algotest::<f64, f64>(&test7_params());
}

#[test]
fn test8() {
    algotest::<f64, f64>(&test8_params());
}

#[test]
fn test9() {
    algotest::<f64, f64>(&test9_params());
}

#[test]
fn test10() {
    algotest::<f64, f64>(&test10_params());
}

#[test]
fn test11() {
    algotest::<f64, f64>(&test11_params());
}

#[test]
fn test12() {
    algotest::<f64, f64>(&test12_params());
}

#[test]
fn test13() {
    algotest::<f64, f64>(&test13_params());
}

#[test]
fn test14() {
    algotest::<f64, f64>(&test14_params());
}

#[test]
fn test15() {
    algotest::<f64, f64>(&test15_params());
}

#[test]
fn test16() {
    algotest::<f64, f64>(&test16_params());
}

#[test]
fn test17() {
    algotest::<f64, f64>(&test17_params());
}

#[test]
fn test18() {
    algotest::<f64, f64>(&test18_params());
}

#[test]
fn test19() {
    algotest::<f64, f64>(&test19_params());
}

#[test]
fn test20() {
    algotest::<f64, f64>(&test20_params());
}

#[test]
fn test21() {
    algotest::<f64, f64>(&test21_params());
}

#[test]
fn test22() {
    algotest::<f64, f64>(&test22_params());
}

#[test]
fn test23() {
    algotest::<f64, f64>(&test23_params());
}

#[test]
fn test24() {
    algotest::<f64, f64>(&test24_params());
}

#[test]
fn test25() {
    algotest::<f64, f64>(&test25_params());
}

#[test]
fn test26() {
    algotest::<f64, f64>(&test26_params());
}

#[test]
fn test27() {
    algotest::<f64, f64>(&test27_params());
}

#[test]
fn test28() {
    algotest::<f64, f64>(&test28_params());
}

#[test]
fn test29() {
    algotest::<f64, f64>(&test29_params());
}

#[test]
#[ignore = "5M atoms"]
fn test30() {
    algotest::<f64, f64>(&test30_params());
}

#[test]
#[ignore = "5M atoms"]
fn test31() {
    algotest::<f64, f64>(&test31_params());
}

#[test]
#[ignore = "5M atoms"]
fn test32() {
    algotest::<f64, f64>(&test32_params());
}

#[test]
#[ignore = "5M atoms"]
fn test33() {
    algotest::<f64, f64>(&test33_params());
}

#[test]
#[ignore = "5M atoms"]
fn test34() {
    algotest::<f64, f64>(&test34_params());
}

#[test]
#[ignore = "5M atoms"]
fn test35() {
    algotest::<f64, f64>(&test35_params());
}

#[test]
#[ignore = "5M atoms"]
fn test36() {
    algotest::<f64, f64>(&test36_params());
}

#[test]
#[ignore = "5M atoms"]
fn test37() {
    algotest::<f64, f64>(&test37_params());
}

#[test]
#[ignore = "5M atoms"]
fn test38() {
    algotest::<f64, f64>(&test38_params());
}

#[test]
#[ignore = "5M atoms"]
fn test39() {
    algotest::<f64, f64>(&test39_params());
}

#[test]
#[ignore = "5M atoms"]
fn test40() {
    algotest::<f64, f64>(&test40_params());
}

#[test]
fn test41() {
    algotest::<f64, f64>(&test41_params());
}

#[test]
fn test42() {
    algotest::<f64, f64>(&test42_params());
}

#[test]
fn test43() {
    algotest::<f64, f64>(&test43_params());
}

#[test]
fn test44() {
    algotest::<f64, f64>(&test44_params());
}

#[test]
fn test45() {
    algotest::<f64, f64>(&test45_params());
}

#[test]
fn test46() {
    algotest::<f64, f64>(&test46_params());
}

#[test]
fn test47() {
    algotest::<f64, f64>(&test47_params());
}

#[test]
fn test48() {
    algotest::<f64, f64>(&test48_params());
}

#[test]
fn test49() {
    algotest::<f64, f64>(&test49_params());
}

#[test]
fn test50() {
    algotest::<f64, f64>(&test50_params());
}

#[test]
fn test51() {
    algotest::<f64, f64>(&test51_params());
}

#[test]
fn test52() {
    algotest::<f64, f64>(&test52_params());
}

#[test]
fn test53() {
    algotest::<f64, f64>(&test53_params());
}

#[test]
fn test54() {
    algotest::<f64, f64>(&test54_params());
}

#[test]
fn test55() {
    algotest::<f64, f64>(&test55_params());
}

#[test]
fn test56() {
    algotest::<f64, f64>(&test56_params());
}

#[test]
fn test57() {
    algotest::<f64, f64>(&test57_params());
}

#[test]
fn test58() {
    algotest::<f32, f32>(&test58_params());
}

#[test]
fn test59() {
    algotest::<f32, f64>(&test59_params());
}
