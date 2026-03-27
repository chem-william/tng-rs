use crate::compress::Float;

/// Deterministic data generators ported from vendor/tng/src/tests/compression/testsuite.c.
/// Used to produce the same input data as the C testsuite for byte-for-byte comparison tests.
///
/// Parameters for a C testsuite test, mirroring the #defines in testN.h.
#[allow(dead_code)]
pub(super) struct TestParams<T: Float> {
    pub natoms: usize,
    pub chunky: usize,
    pub nframes: usize,
    pub scale: f64,
    pub precision: T,
    pub writevel: bool,
    pub velprecision: T,
    pub initial_coding: i32,
    pub initial_coding_parameter: i32,
    pub coding: i32,
    pub coding_parameter: i32,
    pub velcoding: i32,
    pub velcoding_parameter: i32,
    pub initial_velcoding: i32,
    pub initial_velcoding_parameter: i32,
    pub intmin: [i32; 3],
    pub intmax: [i32; 3],
    pub speed: usize,
    pub framescale: i32,
    pub genprecision: f64,
    pub genvelprecision: f64,
    pub expected_filesize: f64,
    pub recompressed_filesize: Option<f64>,
    pub regular: bool,
    pub velintmul: Option<i32>,
    pub recompress: bool,
    pub int_to_double: bool,
}

impl<T: Float> Default for TestParams<T> {
    fn default() -> Self {
        Self {
            natoms: 1000,
            chunky: 100,
            nframes: 1000,
            scale: 0.1,
            precision: T::from_f64(0.01),
            writevel: false,
            velprecision: T::from_f64(0.1),
            initial_coding: 0,
            initial_coding_parameter: 0,
            coding: 0,
            coding_parameter: 0,
            velcoding: 0,
            velcoding_parameter: 0,
            initial_velcoding: 0,
            initial_velcoding_parameter: 0,
            intmin: [0; 3],
            intmax: [0; 3],
            speed: 5,
            framescale: 1,
            genprecision: 0.01,
            genvelprecision: 0.1,
            expected_filesize: 0.0,
            recompressed_filesize: None,
            regular: false,
            velintmul: None,
            recompress: false,
            int_to_double: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Integer sine/cosine lookup table (from testsuite.c lines 42-65)
// ---------------------------------------------------------------------------

const INTSINTABLE: [i32; 128] = [
    0, 3215, 6423, 9615, 12785, 15923, 19023, 22078, 25079, 28019, 30892, 33691, 36409, 39039,
    41574, 44010, 46340, 48558, 50659, 52638, 54490, 56211, 57796, 59242, 60546, 61704, 62713,
    63570, 64275, 64825, 65219, 65456, 65535, 65456, 65219, 64825, 64275, 63570, 62713, 61704,
    60546, 59242, 57796, 56211, 54490, 52638, 50659, 48558, 46340, 44010, 41574, 39039, 36409,
    33691, 30892, 28019, 25079, 22078, 19023, 15923, 12785, 9615, 6423, 3215, 0, -3215, -6423,
    -9615, -12785, -15923, -19023, -22078, -25079, -28019, -30892, -33691, -36409, -39039, -41574,
    -44010, -46340, -48558, -50659, -52638, -54490, -56211, -57796, -59242, -60546, -61704, -62713,
    -63570, -64275, -64825, -65219, -65456, -65535, -65456, -65219, -64825, -64275, -63570, -62713,
    -61704, -60546, -59242, -57796, -56211, -54490, -52638, -50659, -48558, -46340, -44010, -41574,
    -39039, -36409, -33691, -30892, -28019, -25079, -22078, -19023, -15923, -12785, -9615, -6423,
    -3215,
];

/// C: when i < 0, sets i=0 and sign=-1, returns sign * table[0] = 0.
/// Otherwise returns table[i % 128].
fn intsin(i: i32) -> i32 {
    if i < 0 {
        // C sets i=0, sign=-1, returns -1 * intsintable[0] = -1 * 0 = 0
        0
    } else {
        INTSINTABLE[(i % 128) as usize]
    }
}

fn intcos(i: i32) -> i32 {
    let i = if i < 0 { 0 } else { i };
    intsin(i + 32)
}

// ---------------------------------------------------------------------------
// keepinbox (testsuite.c lines 26-40)
// ---------------------------------------------------------------------------

fn keepinbox(val: &mut [i32; 3], intmin: &[i32; 3], intmax: &[i32; 3]) {
    for dim in 0..3 {
        let range = intmax[dim] - intmin[dim] + 1;
        while val[dim] > intmax[dim] {
            val[dim] -= range;
        }
        while val[dim] < intmin[dim] {
            val[dim] += range;
        }
    }
}

// ---------------------------------------------------------------------------
// molecule (testsuite.c lines 74-89)
// ---------------------------------------------------------------------------

fn molecule(
    target: &mut [i32],
    base: &[i32; 3],
    length: usize,
    scale: i32,
    direction: &[i32; 3],
    flip: bool,
    iframe: i32,
    intmin: &[i32; 3],
    intmax: &[i32; 3],
) {
    for i in 0..length {
        let ifl = if flip && length > 1 {
            match i {
                0 => 1,
                1 => 0,
                other => other,
            }
        } else {
            i
        };
        target[ifl * 3] = base[0] + (intsin((i as i32 + iframe) * direction[0]) * scale) / 256;
        target[ifl * 3 + 1] = base[1] + (intcos((i as i32 + iframe) * direction[1]) * scale) / 256;
        target[ifl * 3 + 2] = base[2] + (intcos((i as i32 + iframe) * direction[2]) * scale) / 256;
        let mut atom = [target[ifl * 3], target[ifl * 3 + 1], target[ifl * 3 + 2]];
        keepinbox(&mut atom, intmin, intmax);
        target[ifl * 3] = atom[0];
        target[ifl * 3 + 1] = atom[1];
        target[ifl * 3 + 2] = atom[2];
    }
}

// ---------------------------------------------------------------------------
// genibox (testsuite.c lines 95-168)
// ---------------------------------------------------------------------------

pub(super) fn genibox<T: Float>(intbox: &mut [i32], iframe: i32, params: &TestParams<T>) {
    let natoms = params.natoms;
    let intmin = &params.intmin;
    let intmax = &params.intmax;
    let framescale = params.framescale;

    let mut molecule_length: usize = 1;
    let mut molpos = [
        intsin(iframe * framescale) / 32,
        1 + intcos(iframe * framescale) / 32,
        2 + intsin(iframe * framescale) / 16,
    ];
    keepinbox(&mut molpos, intmin, intmax);

    let mut direction = [1i32, 1, 1];
    let mut scale = 1i32;
    let mut flip = false;
    let mut i: usize = 0;

    while i < natoms {
        let mut this_mol_length = molecule_length;

        if params.regular {
            this_mol_length = 4;
            flip = false;
            scale = 1;
        }

        if i + this_mol_length > natoms {
            this_mol_length = natoms - i;
        }

        if !params.regular && i.is_multiple_of(10) {
            // Large RLE insertion path
            intbox[i * 3] = molpos[0];
            intbox[i * 3 + 1] = molpos[1];
            intbox[i * 3 + 2] = molpos[2];
            for j in 1..this_mol_length {
                intbox[(i + j) * 3] = intbox[(i + j - 1) * 3] + (intmax[0] - intmin[0] + 1) / 5;
                intbox[(i + j) * 3 + 1] =
                    intbox[(i + j - 1) * 3 + 1] + (intmax[1] - intmin[1] + 1) / 5;
                intbox[(i + j) * 3 + 2] =
                    intbox[(i + j - 1) * 3 + 2] + (intmax[2] - intmin[2] + 1) / 5;
                let mut atom = [
                    intbox[(i + j) * 3],
                    intbox[(i + j) * 3 + 1],
                    intbox[(i + j) * 3 + 2],
                ];
                keepinbox(&mut atom, intmin, intmax);
                intbox[(i + j) * 3] = atom[0];
                intbox[(i + j) * 3 + 1] = atom[1];
                intbox[(i + j) * 3 + 2] = atom[2];
            }
        } else {
            molecule(
                &mut intbox[i * 3..],
                &molpos,
                this_mol_length,
                scale,
                &direction,
                flip,
                iframe * framescale,
                intmin,
                intmax,
            );
        }

        i += this_mol_length;

        let dir0 = if intsin(i as i32 * 3) < 0 { -1 } else { 1 };
        molpos[0] += dir0 * (intmax[0] - intmin[0] + 1) / 20;
        let dir1 = if intsin(i as i32 * 5) < 0 { -1 } else { 1 };
        molpos[1] += dir1 * (intmax[1] - intmin[1] + 1) / 20;
        let dir2 = if intsin(i as i32 * 7) < 0 { -1 } else { 1 };
        molpos[2] += dir2 * (intmax[2] - intmin[2] + 1) / 20;
        keepinbox(&mut molpos, intmin, intmax);

        direction[0] = ((direction[0] + 1) % 7) + 1;
        direction[1] = ((direction[1] + 1) % 3) + 1;
        direction[2] = ((direction[2] + 1) % 6) + 1;

        scale += 1;
        if scale > 5 {
            scale = 1;
        }

        molecule_length += 1;
        if molecule_length > 30 {
            molecule_length = 1;
        }
        if !i.is_multiple_of(9) {
            flip = !flip;
        }
    }
}

// ---------------------------------------------------------------------------
// genivelbox (testsuite.c lines 170-188)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub(super) fn genivelbox<T: Float>(intvelbox: &mut [i32], iframe: i32, params: &TestParams<T>) {
    let natoms = params.natoms;
    let framescale = params.framescale;
    for i in 0..natoms {
        let idx = i as i32;
        let f = iframe * framescale;
        if let Some(velintmul) = params.velintmul {
            intvelbox[i * 3] = ((intsin((idx + f) * 3)) / 10) * velintmul + idx;
            intvelbox[i * 3 + 1] = 1 + ((intcos((idx + f) * 5)) / 10) * velintmul + idx;
            intvelbox[i * 3 + 2] =
                2 + ((intsin((idx + f) * 7) + intcos((idx + f) * 9)) / 20) * velintmul + idx;
        } else {
            intvelbox[i * 3] = (intsin((idx + f) * 3)) / 10;
            intvelbox[i * 3 + 1] = 1 + (intcos((idx + f) * 5)) / 10;
            intvelbox[i * 3 + 2] = 2 + (intsin((idx + f) * 7) + intcos((idx + f) * 9)) / 20;
        }
    }
}

// ---------------------------------------------------------------------------
// realbox / realvelbox (testsuite.c lines 206-228)
// ---------------------------------------------------------------------------

pub(super) fn realbox<T: Float>(
    intbox: &[i32],
    pos: &mut [T],
    stride: usize,
    natoms: usize,
    genprecision: f64,
    scale: f64,
) {
    for i in 0..natoms {
        for j in 0..3 {
            pos[i * stride + j] = T::from_f64(intbox[i * 3 + j] as f64 * genprecision * scale);
        }
        for j in 3..stride {
            pos[i * stride + j] = T::from_f64(0.0);
        }
    }
}

#[allow(dead_code)]
pub(super) fn realvelbox<T: Float>(
    intbox: &[i32],
    vel: &mut [T],
    stride: usize,
    natoms: usize,
    genvelprecision: f64,
    scale: f64,
) {
    for i in 0..natoms {
        for j in 0..3 {
            vel[i * stride + j] = T::from_f64(intbox[i * 3 + j] as f64 * genvelprecision * scale);
        }
        for j in 3..stride {
            vel[i * stride + j] = T::from_f64(0.0);
        }
    }
}
