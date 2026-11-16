use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;
use tng_rs::data::{Compression, DataType};
use tng_rs::gen_block::BlockID;
use tng_rs::trajectory::{BlockType, Trajectory};
use tng_rs::{NATOMS, NFRAMES};

/// Writes a `Float` positions trajectory with the shared benchmark workload.
///
/// [`tng_rs::bench_write`] stores `Double` positions, which the `Float`-only readers reject,
/// so the same data is written as `Float` here.
fn write_benchmark_file(dir: &Path) -> PathBuf {
    let path = dir.join("tng_rs_read_criterion.tng");
    let (positions, _velocities) = tng_rs::bench_data();

    let mut trajectory = Trajectory::new();
    trajectory.output_file_set(&path);
    trajectory.compression_precision = 1.0e6;
    trajectory.frame_set_n_frames = NFRAMES as i64;
    trajectory.current_trajectory_frame_set.n_frames = NFRAMES as i64;
    let molecule = trajectory.add_molecule("particle");
    let chain = trajectory.add_chain(molecule, "A");
    let residue = trajectory.chain_residue_add(molecule, chain, "PAR");
    trajectory.residue_atom_add(molecule, residue, "P", "P");
    trajectory.molecule_cnt_set(molecule, NATOMS as i64);
    trajectory.file_headers_write(false).expect("write headers");

    trajectory
        .frame_set_with_time_new(0, NFRAMES as i64, 0.0)
        .expect("create frame set");

    let floats: Vec<f32> = positions.iter().map(|value| *value as f32).collect();
    let bytes: Vec<u8> = floats
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    trajectory
        .particle_data_block_add(
            BlockID::TrajPositions,
            "POSITIONS",
            DataType::Float,
            &BlockType::Trajectory,
            NFRAMES as i64,
            3,
            1,
            0,
            NATOMS as i64,
            Compression::TNG,
            Some(&bytes),
        )
        .expect("add positions");
    trajectory.frame_set_write(false).expect("write frame set");
    trajectory.trajectory_destroy().expect("close trajectory");

    path
}

fn open(path: &Path) -> Trajectory {
    let mut trajectory = Trajectory::new();
    trajectory.input_file_set(path);
    trajectory.file_headers_read(true).expect("read headers");
    trajectory
}

/// Compares the `f64` positions reader with the `f32` reader on the same file
/// and workload.
fn bench_reads(c: &mut Criterion) {
    let dir = TempDir::new().expect("create temporary directory");
    let path = write_benchmark_file(dir.path());
    let last_frame = NFRAMES as i64 - 1;

    c.bench_function("read_positions_f64", |b| {
        b.iter_batched(
            || open(&path),
            |mut trajectory| {
                let (values, stride) = trajectory
                    .util_pos_read_range(0, last_frame)
                    .expect("read f64 positions");
                black_box((values.len() * std::mem::size_of::<f64>(), stride))
            },
            BatchSize::LargeInput,
        );
    });

    c.bench_function("read_positions_f32", |b| {
        b.iter_batched(
            || open(&path),
            |mut trajectory| {
                let (values, stride) = trajectory
                    .util_pos_read_range_f32(0, last_frame)
                    .expect("read f32 positions");
                black_box((values.len() * std::mem::size_of::<f32>(), stride))
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets = bench_reads
}
criterion_main!(benches);
