use crate::{
    TngError,
    data::{Compression, DataType},
    gen_block::BlockID,
    trajectory::{BlockType, Trajectory},
};
use std::path::Path;
use zerocopy::IntoBytes;

const NATOMS: usize = 100_000;
const NFRAMES: usize = 20;
const CHUNKY: usize = 10;
const SCALE: f64 = 0.5;
const PRECISION: f64 = 1e-8;
const VELINTMUL: i32 = 100_000;
const TIME_PER_FRAME: f64 = 2e-15;
const INTMIN: [i32; 3] = [0; 3];
const INTMAX: [i32; 3] = [805_306_368; 3];

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

/// Generate benchmark position and velocity data.
pub fn bench_data() -> (Vec<f64>, Vec<f64>) {
    let frame_len = NATOMS * 3;
    let mut positions = vec![0.0; NFRAMES * frame_len];
    let mut velocities = vec![0.0; NFRAMES * frame_len];
    let mut intbox = vec![0; frame_len];
    let mut intvelbox = vec![0; frame_len];

    for frame in 0..NFRAMES {
        genibox(&mut intbox, frame as i32);
        genivelbox(&mut intvelbox, frame as i32);

        let start = frame * frame_len;
        let end = start + frame_len;
        realbox(&intbox, &mut positions[start..end]);
        realbox(&intvelbox, &mut velocities[start..end]);
    }

    (positions, velocities)
}

/// Write benchmark data to a TNG file. Returns file size in bytes.
pub fn bench_write(path: &Path, positions: &[f64], velocities: &[f64]) -> Result<u64, TngError> {
    let frame_len = NATOMS * 3;
    let expected_len = NFRAMES * frame_len;
    assert_eq!(positions.len(), expected_len);
    assert_eq!(velocities.len(), expected_len);

    let mut traj = Trajectory::new();
    traj.output_file_set(path);
    traj.frame_set_n_frames = CHUNKY as i64;
    traj.compression_precision = 1.0 / PRECISION;
    traj.set_time_per_frame(TIME_PER_FRAME)?;

    let molecule_idx = traj.add_molecule("particle");
    let chain_idx = traj.add_chain(molecule_idx, "A");
    let residue_idx = traj.chain_residue_add(molecule_idx, chain_idx, "PAR");
    traj.residue_atom_add(molecule_idx, residue_idx, "P", "P");
    traj.molecule_cnt_set(molecule_idx, NATOMS as i64);

    traj.file_headers_write(false)?;

    for frame_start in (0..NFRAMES).step_by(CHUNKY) {
        let frames_in_chunk = (NFRAMES - frame_start).min(CHUNKY);
        let first_frame = frame_start as i64;
        let time = first_frame as f64 * TIME_PER_FRAME;
        let start = frame_start * frame_len;
        let end = start + frames_in_chunk * frame_len;

        traj.frame_set_with_time_new(first_frame, frames_in_chunk as i64, time)?;

        let pos_bytes = positions[start..end].as_bytes();
        traj.particle_data_block_add(
            BlockID::TrajPositions,
            "POSITIONS",
            DataType::Double,
            &BlockType::Trajectory,
            frames_in_chunk as i64,
            3,
            1,
            0,
            NATOMS as i64,
            Compression::TNG,
            Some(pos_bytes),
        )?;

        let vel_bytes = velocities[start..end].as_bytes();
        traj.particle_data_block_add(
            BlockID::TrajVelocities,
            "VELOCITIES",
            DataType::Double,
            &BlockType::Trajectory,
            frames_in_chunk as i64,
            3,
            1,
            0,
            NATOMS as i64,
            Compression::TNG,
            Some(vel_bytes),
        )?;

        traj.frame_set_write(false)?;
    }

    traj.trajectory_destroy()?;
    Ok(std::fs::metadata(path)?.len())
}

// --- data generation helpers ---

fn intsin(i: i32) -> i32 {
    if i < 0 {
        0
    } else {
        INTSINTABLE[(i % 128) as usize]
    }
}

fn intcos(i: i32) -> i32 {
    intsin(if i < 0 { 0 } else { i } + 32)
}

fn keepinbox(val: &mut [i32; 3]) {
    for dim in 0..3 {
        let range = INTMAX[dim] - INTMIN[dim] + 1;
        while val[dim] > INTMAX[dim] {
            val[dim] -= range;
        }
        while val[dim] < INTMIN[dim] {
            val[dim] += range;
        }
    }
}

fn molecule(
    target: &mut [i32],
    base: [i32; 3],
    length: usize,
    scale: i32,
    direction: [i32; 3],
    flip: bool,
    iframe: i32,
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
        keepinbox(&mut atom);
        target[ifl * 3] = atom[0];
        target[ifl * 3 + 1] = atom[1];
        target[ifl * 3 + 2] = atom[2];
    }
}

fn genibox(intbox: &mut [i32], iframe: i32) {
    let mut molecule_length = 1usize;
    let mut molpos = [
        intsin(iframe) / 32,
        1 + intcos(iframe) / 32,
        2 + intsin(iframe) / 16,
    ];
    keepinbox(&mut molpos);

    let mut direction = [1, 1, 1];
    let mut scale = 1;
    let mut flip = false;
    let mut i = 0usize;

    while i < NATOMS {
        let this_mol_length = molecule_length.min(NATOMS - i);

        if i.is_multiple_of(10) {
            intbox[i * 3] = molpos[0];
            intbox[i * 3 + 1] = molpos[1];
            intbox[i * 3 + 2] = molpos[2];
            for j in 1..this_mol_length {
                intbox[(i + j) * 3] = intbox[(i + j - 1) * 3] + (INTMAX[0] - INTMIN[0] + 1) / 5;
                intbox[(i + j) * 3 + 1] =
                    intbox[(i + j - 1) * 3 + 1] + (INTMAX[1] - INTMIN[1] + 1) / 5;
                intbox[(i + j) * 3 + 2] =
                    intbox[(i + j - 1) * 3 + 2] + (INTMAX[2] - INTMIN[2] + 1) / 5;
                let mut atom = [
                    intbox[(i + j) * 3],
                    intbox[(i + j) * 3 + 1],
                    intbox[(i + j) * 3 + 2],
                ];
                keepinbox(&mut atom);
                intbox[(i + j) * 3] = atom[0];
                intbox[(i + j) * 3 + 1] = atom[1];
                intbox[(i + j) * 3 + 2] = atom[2];
            }
        } else {
            molecule(
                &mut intbox[i * 3..],
                molpos,
                this_mol_length,
                scale,
                direction,
                flip,
                iframe,
            );
        }

        i += this_mol_length;

        let dir0 = if intsin(i as i32 * 3) < 0 { -1 } else { 1 };
        molpos[0] += dir0 * (INTMAX[0] - INTMIN[0] + 1) / 20;
        let dir1 = if intsin(i as i32 * 5) < 0 { -1 } else { 1 };
        molpos[1] += dir1 * (INTMAX[1] - INTMIN[1] + 1) / 20;
        let dir2 = if intsin(i as i32 * 7) < 0 { -1 } else { 1 };
        molpos[2] += dir2 * (INTMAX[2] - INTMIN[2] + 1) / 20;
        keepinbox(&mut molpos);

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

fn genivelbox(intvelbox: &mut [i32], iframe: i32) {
    for i in 0..NATOMS {
        let idx = i as i32;
        intvelbox[i * 3] = (intsin((idx + iframe) * 3) / 10) * VELINTMUL + idx;
        intvelbox[i * 3 + 1] = 1 + (intcos((idx + iframe) * 5) / 10) * VELINTMUL + idx;
        intvelbox[i * 3 + 2] =
            2 + ((intsin((idx + iframe) * 7) + intcos((idx + iframe) * 9)) / 20) * VELINTMUL + idx;
    }
}

fn realbox(intbox: &[i32], out: &mut [f64]) {
    for i in 0..NATOMS {
        for j in 0..3 {
            out[i * 3 + j] = f64::from(intbox[i * 3 + j]) * PRECISION * SCALE;
        }
    }
}
