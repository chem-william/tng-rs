use std::ffi::{c_int, c_ulong};

use crate::ffi;

/// Copy `nitems` bytes from a C-allocated pointer into a `Vec<u8>`, then free the pointer.
///
/// # Safety
/// `ptr` must be a valid, non-null pointer to at least `nitems` bytes allocated by C `malloc`.
pub(super) unsafe fn copy_and_free(ptr: *mut i8, nitems: c_int) -> Vec<u8> {
    assert!(!ptr.is_null(), "C compress function returned NULL");
    let len = nitems as usize;
    let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, len).to_vec() };
    unsafe { libc::free(ptr as *mut libc::c_void) };
    data
}

// ---------------------------------------------------------------------------
// C compress wrappers
// ---------------------------------------------------------------------------

pub(super) fn c_compress_pos_int(
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

pub(super) fn c_compress_pos(
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

pub(super) fn c_compress_pos_float(
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

pub(super) fn c_compress_vel_int(
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

pub(super) fn c_compress_vel(
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

pub(super) fn c_compress_vel_float(
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

pub(super) fn c_compress_pos_find_algo(
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
// C decompress wrappers
// ---------------------------------------------------------------------------

/// Decompress a compressed buffer using C's `tng_compress_uncompress`.
/// Returns the decompressed position/velocity data as `Vec<f64>`.
/// `n_values` is the total number of doubles expected (natoms * nframes * 3).
pub(super) fn c_uncompress(compressed: &[u8], n_values: usize) -> Vec<f64> {
    let mut buf = compressed.iter().map(|&b| b as i8).collect::<Vec<i8>>();
    let mut output = vec![0.0f64; n_values];
    let ret = unsafe {
        ffi::tng_compress_uncompress(buf.as_mut_ptr(), output.as_mut_ptr())
    };
    assert_eq!(ret, 0, "C tng_compress_uncompress returned error {}", ret);
    output
}
