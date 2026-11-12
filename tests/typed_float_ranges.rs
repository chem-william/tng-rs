//! Tests for the native `f32` range readers (`util_*_read_range_f32`).
//!
//! Expected values are computed structurally from the block layout the writer produces
//! (frame-major, then particle-major, then components).
//! `legacy_f64_readers_match_known_values` checks the legacy readers against the
//! same independent expectations, so a bug shared by both implementations cannot pass.

use std::fs;

use tng_rs::TngError;
use tng_rs::data::{Compression, DataType};
use tng_rs::gen_block::BlockID;
use tng_rs::trajectory::{BlockType, Trajectory};

const PARTICLES: i64 = 4;
const VALUES_PER_FRAME: i64 = 3;
const BOX_VALUES: i64 = 9;

/// Deterministic value for a stored sample / particle / component triple.
///
/// Magnitudes stay well inside the precision of TNG lossy compression, and distinct samples
/// produce distinct values so a missing or duplicated sample is detected.
fn sample_value(stored_frame: i64, particle: i64, component: i64) -> f32 {
    1.0 + (stored_frame as f32) * 100.0 + (particle as f32) * 10.0 + (component as f32) * 0.5
}

/// Deterministic box-shape value for a stored sample / component pair.
fn box_value(stored_frame: i64, component: i64) -> f32 {
    50.0 + (stored_frame as f32) * 100.0 + (component as f32) * 0.25
}

fn native_f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// Byte offset added to each stored value so every block has a distinguishable payload.
const VELOCITY_OFFSET: f32 = 1000.0;
const FORCE_OFFSET: f32 = 2000.0;

/// Encodes `n_stored_frames` stored samples of particle data, starting at global stored
/// sample `first_stored`, in `data_type`'s byte representation.
fn particle_payload(
    data_type: DataType,
    offset: f32,
    first_stored: i64,
    n_stored_frames: i64,
) -> Vec<u8> {
    let mut values = Vec::new();
    for stored_frame in first_stored..first_stored + n_stored_frames {
        for particle in 0..PARTICLES {
            for component in 0..VALUES_PER_FRAME {
                let value = sample_value(stored_frame, particle, component) + offset;
                match data_type {
                    DataType::Float => values.extend_from_slice(&value.to_ne_bytes()),
                    DataType::Double => {
                        values.extend_from_slice(&f64::from(value).to_ne_bytes());
                    }
                    DataType::Int => {
                        values.extend_from_slice(&(value as i64).to_ne_bytes());
                    }
                    DataType::Char => panic!("Char payloads are handled as strings"),
                }
            }
        }
    }
    values
}

fn box_payload(first_stored: i64, n_stored_frames: i64) -> Vec<u8> {
    let mut values = Vec::new();
    for stored_frame in first_stored..first_stored + n_stored_frames {
        for component in 0..BOX_VALUES {
            values.push(box_value(stored_frame, component));
        }
    }
    native_f32_bytes(&values)
}

struct Spec {
    frames_per_set: i64,
    stride: i64,
    compression: Compression,
    frame_sets: i64,
    /// Data type used for the positions block.
    positions_type: DataType,
    /// Whether every frame set maps its local particles to [`real_particle`] numbers.
    mapping: bool,
}

impl Spec {
    fn new() -> Self {
        Self {
            frames_per_set: 1,
            stride: 1,
            compression: Compression::Uncompressed,
            frame_sets: 1,
            positions_type: DataType::Float,
            mapping: false,
        }
    }

    /// Particle number under which the readers return local particle `particle`.
    fn returned_particle(&self, particle: i64) -> i64 {
        if self.mapping {
            real_particle(particle)
        } else {
            particle
        }
    }

    /// Number of samples stored per frame set.
    fn stored_frames_per_set(&self) -> i64 {
        (self.frames_per_set - 1) / self.stride + 1
    }

    /// First global stored-sample index of frame set `frame_set`.
    ///
    /// Only meaningful when `frames_per_set` is a multiple of `stride` (every test with more
    /// than one frame set satisfies this), which keeps the frame-set-local sample grid aligned
    /// with the global `frame / stride` indexing the readers use.
    fn first_stored_of_frame_set(&self, frame_set: i64) -> i64 {
        frame_set * self.frames_per_set / self.stride
    }
}

/// Real particle number of local particle `particle` in a mapped frame set: the order is
/// reversed, so a reader that ignores the mapping returns the particles in the wrong slots.
fn real_particle(particle: i64) -> i64 {
    PARTICLES - 1 - particle
}

/// Adds one frame set worth of blocks to `trajectory`.
fn write_frame_set(trajectory: &mut Trajectory, spec: &Spec, frame_set: i64) {
    let first_frame = frame_set * spec.frames_per_set;
    trajectory
        .frame_set_with_time_new(first_frame, spec.frames_per_set, first_frame as f64 * 2e-15)
        .expect("create frame set");

    if spec.mapping {
        // A new frame set keeps the mappings of the previous one.
        trajectory.frame_set_particle_mapping_free();
        let real_particles: Vec<i64> = (0..PARTICLES).map(real_particle).collect();
        trajectory
            .particle_mapping_add(0, PARTICLES, &real_particles)
            .expect("add particle mapping");
    }

    let n_stored = spec.stored_frames_per_set();
    let first_stored = spec.first_stored_of_frame_set(frame_set);
    for (id, name, offset) in [
        (BlockID::TrajPositions, "POSITIONS", 0.0),
        (BlockID::TrajVelocities, "VELOCITIES", VELOCITY_OFFSET),
        (BlockID::TrajForces, "FORCES", FORCE_OFFSET),
    ] {
        let data_type = if id == BlockID::TrajPositions {
            spec.positions_type
        } else {
            DataType::Float
        };
        let payload = particle_payload(data_type, offset, first_stored, n_stored);
        // TNG compression only supports positions and velocities.
        let compression = if id == BlockID::TrajPositions || spec.compression != Compression::TNG {
            spec.compression
        } else {
            Compression::GZip
        };
        trajectory
            .particle_data_block_add(
                id,
                name,
                data_type,
                &BlockType::Trajectory,
                spec.frames_per_set,
                VALUES_PER_FRAME,
                spec.stride,
                0,
                PARTICLES,
                compression,
                Some(&payload),
            )
            .expect("add particle block");
    }

    trajectory
        .data_block_add(
            BlockID::TrajBoxShape,
            "BOX SHAPE",
            DataType::Float,
            &BlockType::Trajectory,
            spec.frames_per_set,
            BOX_VALUES,
            spec.stride,
            // TNG compression only supports positions and velocities.
            if spec.compression == Compression::TNG {
                Compression::GZip
            } else {
                spec.compression
            },
            Some(&box_payload(first_stored, n_stored)),
        )
        .expect("add box block");

    trajectory.frame_set_write(true).expect("write frame set");
}

fn write_trajectory(name: &str, spec: &Spec) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join(format!("{name}.tng"));

    let mut trajectory = Trajectory::new();
    trajectory.output_file_set(&path);
    // Values reach a few hundred, so 1e6 keeps the quantiser inside the i32 range.
    trajectory.compression_precision = 1.0e6;
    let molecule = trajectory.add_molecule("particles");
    let chain = trajectory.add_chain(molecule, "A");
    let residue = trajectory.chain_residue_add(molecule, chain, "RES");
    for _ in 0..PARTICLES {
        trajectory.residue_atom_add(molecule, residue, "P", "P");
    }
    trajectory.molecule_cnt_set(molecule, 1);
    trajectory.file_headers_write(true).expect("write headers");

    for frame_set in 0..spec.frame_sets {
        write_frame_set(&mut trajectory, spec, frame_set);
    }

    trajectory.trajectory_destroy().expect("close trajectory");
    (directory, path)
}

fn open_trajectory(path: &std::path::Path) -> Trajectory {
    let mut trajectory = Trajectory::new();
    trajectory.input_file_set(path);
    trajectory.file_headers_read(true).expect("read headers");
    trajectory
}

/// Global index of the stored sample that supplies output sample `sample` of a range that
/// starts at `first_frame`.
///
/// Sparse blocks index their storage by the *global* frame (`frame / stride`), and the first
/// output slot of a range is always the stored sample containing `first_frame`; later slots
/// follow the stride grid.
fn stored_index_of_sample(first_frame: i64, stride: i64, sample: i64) -> i64 {
    first_frame / stride + sample
}

/// Number of samples a range of `first_frame..=last_frame` yields for the given stride.
fn n_samples_in_range(first_frame: i64, last_frame: i64, stride: i64) -> i64 {
    (last_frame - first_frame) / stride + 1
}

/// Expected legacy (`f64`) values of the particle readers for a range.
fn expected_particle_values(
    spec: &Spec,
    offset: f32,
    first_frame: i64,
    last_frame: i64,
) -> Vec<f32> {
    let n_samples = n_samples_in_range(first_frame, last_frame, spec.stride);
    let mut expected = vec![0.0f32; (n_samples * PARTICLES * VALUES_PER_FRAME) as usize];
    for sample in 0..n_samples {
        let stored = stored_index_of_sample(first_frame, spec.stride, sample);
        for particle in 0..PARTICLES {
            let slot = sample * PARTICLES + spec.returned_particle(particle);
            for component in 0..VALUES_PER_FRAME {
                expected[(slot * VALUES_PER_FRAME + component) as usize] =
                    sample_value(stored, particle, component) + offset;
            }
        }
    }
    expected
}

fn expected_box_values(spec: &Spec, first_frame: i64, last_frame: i64) -> Vec<f32> {
    let n_samples = n_samples_in_range(first_frame, last_frame, spec.stride);
    let mut expected = vec![0.0f32; (n_samples * BOX_VALUES) as usize];
    for sample in 0..n_samples {
        let stored = stored_index_of_sample(first_frame, spec.stride, sample);
        for component in 0..BOX_VALUES {
            expected[(sample * BOX_VALUES + component) as usize] = box_value(stored, component);
        }
    }
    expected
}

/// Tolerance per compression mode.
fn tolerance(compression: Compression) -> f32 {
    match compression {
        // Uncompressed and GZip payloads must round-trip bit-exactly.
        Compression::Uncompressed | Compression::GZip => 0.0,
        // TNG is quantised with a multiplier of 1e6.
        Compression::TNG => 1e-3,
        Compression::XTC => 1e-2,
    }
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: unexpected number of values"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "{context}: value {index} was {actual}, expected {expected} (tolerance {tol})"
        );
    }
}

/// Reads every `util_*_read_range_f32` over the full trajectory and checks the exact values.
fn check_particle_reads(spec: &Spec, name: &str) {
    let (_directory, path) = write_trajectory(name, spec);
    let last_frame = spec.frames_per_set * spec.frame_sets - 1;
    let tol = tolerance(spec.compression);
    let expected_box = expected_box_values(spec, 0, last_frame);

    let mut trajectory = open_trajectory(&path);
    for (reader, offset, label) in [
        (
            Trajectory::util_pos_read_range_f32
                as fn(&mut Trajectory, i64, i64) -> Result<(Vec<f32>, i64), TngError>,
            0.0,
            "positions",
        ),
        (
            Trajectory::util_vel_read_range_f32,
            VELOCITY_OFFSET,
            "velocities",
        ),
        (
            Trajectory::util_force_read_range_f32,
            FORCE_OFFSET,
            "forces",
        ),
    ] {
        let (values, stride) = reader(&mut trajectory, 0, last_frame)
            .unwrap_or_else(|error| panic!("{name}/{label}: {error}"));
        assert_eq!(stride, spec.stride, "{name}/{label}: stride mismatch");
        assert_close_f32(
            &values,
            &expected_particle_values(spec, offset, 0, last_frame),
            tol,
            &format!("{name}/{label}"),
        );
    }

    let (box_shape, stride) = trajectory
        .util_box_shape_read_range_f32(0, last_frame)
        .unwrap_or_else(|error| panic!("{name}/box: {error}"));
    assert_eq!(stride, spec.stride, "{name}/box: stride mismatch");
    assert_close_f32(&box_shape, &expected_box, tol, &format!("{name}/box"));
}

#[test]
fn uncompressed_f32_reads_exact_values() {
    check_particle_reads(&Spec::new(), "uncompressed");
}

#[test]
fn gzip_f32_reads_exact_values() {
    check_particle_reads(
        &Spec {
            compression: Compression::GZip,
            ..Spec::new()
        },
        "gzip",
    );
}

#[test]
fn tng_f32_reads_within_compression_precision() {
    // TNG compression requires more than a single frame in this writer, so use a sparse
    // multi-frame frame set (the TNG payload round-trips through the quantiser).
    check_particle_reads(
        &Spec {
            frames_per_set: 10,
            stride: 5,
            compression: Compression::TNG,
            ..Spec::new()
        },
        "tng",
    );
}

#[test]
fn all_four_util_apis_exact_values() {
    let spec = Spec::new();
    let (_directory, path) = write_trajectory("all-four", &spec);
    let mut trajectory = open_trajectory(&path);

    let (positions, stride) = trajectory.util_pos_read_range_f32(0, 0).expect("positions");
    assert_eq!(stride, 1);
    assert_eq!(
        positions,
        vec![
            1.0, 1.5, 2.0, // particle 0
            11.0, 11.5, 12.0, // particle 1
            21.0, 21.5, 22.0, // particle 2
            31.0, 31.5, 32.0, // particle 3
        ]
    );

    // Each block carries a distinguishable payload; the reader must not mix them up.
    let (velocities, _) = trajectory
        .util_vel_read_range_f32(0, 0)
        .expect("velocities");
    let expected_velocities: Vec<f32> = positions.iter().map(|v| v + VELOCITY_OFFSET).collect();
    assert_eq!(velocities, expected_velocities);

    let (forces, _) = trajectory.util_force_read_range_f32(0, 0).expect("forces");
    let expected_forces: Vec<f32> = positions.iter().map(|v| v + FORCE_OFFSET).collect();
    assert_eq!(forces, expected_forces);

    let (box_shape, stride) = trajectory
        .util_box_shape_read_range_f32(0, 0)
        .expect("box shape");
    assert_eq!(stride, 1);
    let expected: Vec<f32> = (0..BOX_VALUES)
        .map(|component| box_value(0, component))
        .collect();
    assert_eq!(box_shape, expected);
}

#[test]
fn multiple_frame_sets_cross_the_boundary() {
    let spec = Spec {
        frames_per_set: 10,
        stride: 5,
        frame_sets: 2,
        ..Spec::new()
    };
    check_particle_reads(&spec, "multi-frame-set");

    let (_directory, path) = write_trajectory("multi-frame-set-order", &spec);
    let mut trajectory = open_trajectory(&path);
    // Frames 0..=19 span both frame sets and both stored samples of each.
    let (values, stride) = trajectory
        .util_pos_read_range_f32(0, 19)
        .expect("positions");
    assert_eq!(stride, 5);
    let expected = expected_particle_values(&spec, 0.0, 0, 19);
    assert_close_f32(&values, &expected, 0.0, "multi-frame-set-order");

    // Every expected sample must be present exactly once, in order, and nothing must be
    // duplicated across the frame-set boundary.
    let frame_size = (PARTICLES * VALUES_PER_FRAME) as usize;
    assert_eq!(values.len(), 4 * frame_size);
    for sample in 0..4i64 {
        assert_eq!(
            values[sample as usize * frame_size],
            sample_value(sample, 0, 0),
            "sample {sample} was not decoded in order"
        );
    }

    // A partial range inside the second frame set must not touch the first one.
    let (values, _) = trajectory
        .util_pos_read_range_f32(12, 19)
        .expect("positions of the second frame set");
    assert_eq!(values.len(), 2 * frame_size);
    assert_eq!(values[0], sample_value(2, 0, 0));
    assert_eq!(values[frame_size], sample_value(3, 0, 0));
}

#[test]
fn mapping_reorders_into_real_particle_numbering() {
    let spec = Spec {
        frames_per_set: 4,
        mapping: true,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("mapping", &spec);
    let mut trajectory = open_trajectory(&path);

    let (values, stride) = trajectory.util_pos_read_range_f32(0, 3).unwrap();
    assert_eq!(stride, 1);
    assert_eq!(values, expected_particle_values(&spec, 0.0, 0, 3));
    // Local particle 0 (value 1.0) is stored under real particle 3.
    assert_eq!(
        values[(3 * VALUES_PER_FRAME) as usize],
        sample_value(0, 0, 0)
    );
}

#[test]
fn mapping_with_sparse_stride_across_frame_sets() {
    let spec = Spec {
        frames_per_set: 10,
        stride: 5,
        frame_sets: 2,
        mapping: true,
        ..Spec::new()
    };
    check_particle_reads(&spec, "mapping-sparse");

    let (_directory, path) = write_trajectory("mapping-sparse-positions", &spec);
    let mut trajectory = open_trajectory(&path);
    let (values, stride) = trajectory.util_pos_read_range_f32(0, 19).unwrap();
    assert_eq!(stride, 5);
    assert_eq!(values.len(), 4 * (PARTICLES * VALUES_PER_FRAME) as usize);
    assert_eq!(values, expected_particle_values(&spec, 0.0, 0, 19));
}

#[test]
fn multi_frame_set_range_beyond_the_last_frame_is_rejected() {
    let spec = Spec {
        frames_per_set: 10,
        stride: 5,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("past-end", &spec);
    let mut trajectory = open_trajectory(&path);
    assert!(trajectory.util_pos_read_range_f32(0, 9).is_ok());
    // The trajectory stores 10 frames; asking beyond it must be an error, not a panic.
    assert!(trajectory.util_pos_read_range_f32(0, 40).is_err());
}

#[test]
fn partial_ranges_within_sparse_block() {
    // 10 logical frames, one stored sample every 5 frames => samples at frames 0 and 5.
    let spec = Spec {
        frames_per_set: 10,
        stride: 5,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("partial-ranges", &spec);
    let mut trajectory = open_trajectory(&path);
    let frame_size = (PARTICLES * VALUES_PER_FRAME) as usize;

    // Exact first stored sample.
    let (values, stride) = trajectory.util_pos_read_range_f32(0, 0).unwrap();
    assert_eq!(stride, 5);
    assert_eq!(values, expected_particle_values(&spec, 0.0, 0, 0));

    // Whole first stride window, still one stored sample.
    let (values, _) = trajectory.util_pos_read_range_f32(0, 4).unwrap();
    assert_eq!(values.len(), frame_size);
    assert_eq!(values, expected_particle_values(&spec, 0.0, 0, 4));

    // Range starting between stored samples but reaching the next one.
    let (values, _) = trajectory.util_pos_read_range_f32(2, 9).unwrap();
    assert_eq!(values.len(), 2 * frame_size);
    // The range starts inside the window of stored sample 0, so that sample is returned first.
    assert_eq!(values[0], sample_value(0, 0, 0));
    assert_eq!(values[frame_size], sample_value(1, 0, 0));

    // Single frame sitting exactly on the sparse grid (frame 5 is stored sample 1).
    let (values, _) = trajectory.util_pos_read_range_f32(5, 5).unwrap();
    assert_eq!(values.len(), frame_size);
    assert_eq!(values[0], sample_value(1, 0, 0));
    // A single frame inside the second stride window belongs to stored sample 1.
    let (values, _) = trajectory.util_pos_read_range_f32(6, 6).unwrap();
    assert_eq!(values.len(), frame_size);
    assert_eq!(values[0], sample_value(1, 0, 0));

    // Range ending between stored samples.
    let (values, _) = trajectory.util_pos_read_range_f32(5, 7).unwrap();
    assert_eq!(values.len(), frame_size);
    assert_eq!(values[0], sample_value(1, 0, 0));

    // A range that starts inside a stride window and ends before the next stored sample
    // cannot reach any data for this block.
    let error = trajectory
        .util_pos_read_range_f32(1, 3)
        .expect_err("range does not reach the next stored sample");
    assert!(
        matches!(error, TngError::Constraint(_)),
        "unexpected error: {error:?}"
    );
}

#[test]
fn sparse_stride_uses_stored_frames_not_logical_frames() {
    // The scenario called out in review: 6 logical frames with stride 5 store exactly 2 samples.
    let spec = Spec {
        frames_per_set: 6,
        stride: 5,
        ..Spec::new()
    };
    assert_eq!(spec.stored_frames_per_set(), 2);

    let (_directory, path) = write_trajectory("sparse-stride", &spec);
    let mut trajectory = open_trajectory(&path);
    let frame_size = (PARTICLES * VALUES_PER_FRAME) as usize;

    let (values, stride) = trajectory.util_pos_read_range_f32(0, 5).unwrap();
    assert_eq!(stride, 5);
    assert_eq!(
        values.len(),
        2 * frame_size,
        "only the two stored samples may be returned"
    );
    assert_eq!(values, expected_particle_values(&spec, 0.0, 0, 5));

    let (box_shape, _) = trajectory.util_box_shape_read_range_f32(0, 5).unwrap();
    assert_eq!(box_shape.len(), (2 * BOX_VALUES) as usize);
    assert_eq!(box_shape, expected_box_values(&spec, 0, 5));
}

#[test]
fn wrong_data_type_reports_typed_error() {
    let spec = Spec {
        positions_type: DataType::Double,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("wrong-data-type", &spec);
    let mut trajectory = open_trajectory(&path);

    let error = trajectory
        .util_pos_read_range_f32(0, 0)
        .expect_err("reading Double data as f32 must fail");
    assert!(
        matches!(&error, TngError::Constraint(message) if message.contains("as f32")),
        "unexpected error: {error:?}"
    );

    // The legacy f64 range API also rejects non-Float blocks, with its own message; the new
    // API must not regress that contract.
    let legacy = trajectory
        .util_pos_read_range(0, 0)
        .expect_err("legacy range API only supports Float blocks");
    assert!(
        matches!(&legacy, TngError::Constraint(message) if message.contains("not float")),
        "unexpected legacy error: {legacy:?}"
    );
}

#[test]
#[should_panic(expected = "must be lower or equal to")]
fn reversed_range_panics() {
    let spec = Spec::new();
    let (_directory, path) = write_trajectory("reversed-range", &spec);
    let mut trajectory = open_trajectory(&path);
    let _ = trajectory.util_pos_read_range_f32(1, 0);
}

#[test]
fn legacy_f64_readers_match_known_values() {
    let spec = Spec {
        frames_per_set: 10,
        stride: 5,
        frame_sets: 2,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("legacy-f64", &spec);
    let mut trajectory = open_trajectory(&path);

    for (values, offset, label) in [
        (
            trajectory.util_pos_read_range(0, 19).unwrap(),
            0.0,
            "legacy positions",
        ),
        (
            trajectory.util_vel_read_range(0, 19).unwrap(),
            VELOCITY_OFFSET,
            "legacy velocities",
        ),
        (
            trajectory.util_force_read_range(0, 19).unwrap(),
            FORCE_OFFSET,
            "legacy forces",
        ),
    ] {
        let (values, stride) = values;
        assert_eq!(stride, 5, "{label}: stride mismatch");
        assert_close_f32(
            &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            &expected_particle_values(&spec, offset, 0, 19),
            0.0,
            label,
        );
    }

    let (box_shape, stride) = trajectory.util_box_shape_read_range(0, 19).unwrap();
    assert_eq!(stride, 5);
    assert_close_f32(
        &box_shape
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        &expected_box_values(&spec, 0, 19),
        0.0,
        "legacy box shape",
    );
}

#[test]
fn corrupt_payload_is_reported_without_aborting() {
    // Two frame sets: re-reading frame set 0 after frame set 1 forces the block to be loaded
    // from the file a second time, which is the only way the "count every block once" rule in
    // `record_hash_mismatch` is actually exercised. With a single frame set the block stays
    // cached and the second read never re-reads it, so the assertion below would hold trivially.
    let spec = Spec {
        frame_sets: 2,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("corrupt-payload", &spec);

    // Locate the *positions* payload structurally: the four-byte length of the block contents
    // precedes the header's trailing name, then the payload follows. Instead of guessing, walk
    // the block headers and remember the contents offset of the `TrajPositions` block.
    let mut bytes = fs::read(&path).expect("read trajectory");
    let contents =
        find_block_contents(&bytes, BlockID::TrajPositions).expect("positions block must exist");

    // The block contents begin with the block meta-information, so locate the payload inside
    // the known contents region rather than guessing a fixed prefix length.
    let payload = expected_particle_values(&spec, 0.0, 0, 0);
    let payload_bytes = native_f32_bytes(&payload);
    let matches: Vec<usize> = bytes[contents.clone()]
        .windows(payload_bytes.len())
        .enumerate()
        .filter(|(_, window)| *window == payload_bytes.as_slice())
        .map(|(index, _)| contents.start + index)
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "the positions payload must appear exactly once in its own block contents, got {matches:?}"
    );
    let positions_offset = matches[0];

    // Corrupt the third value (a component of particle 0) and leave its neighbours intact.
    let corrupt_at = positions_offset + 2 * std::mem::size_of::<f32>();
    bytes[corrupt_at] ^= 0xff;
    fs::write(&path, bytes).expect("write tampered trajectory");

    let mut trajectory = open_trajectory(&path);
    let (values, _) = trajectory
        .util_pos_read_range_f32(0, 0)
        .expect("a hash mismatch must not abort the read");
    assert_ne!(values[2], payload[2], "the corrupted value must be visible");
    assert_eq!(values[0], payload[0], "neighbouring values are untouched");
    assert_eq!(values[1], payload[1]);
    assert_eq!(values[3], payload[3]);

    assert_eq!(trajectory.hash_mismatch_count(), 1);
    assert_eq!(trajectory.mismatched_block_ids(), [BlockID::TrajPositions]);

    // Switch to frame set 1 and back. Re-reading frame set 0 reloads the corrupted block from
    // the file, so without the dedupe in `record_hash_mismatch` the count would climb to 2 and
    // the id list would hold a second entry.
    let _ = trajectory
        .util_pos_read_range_f32(1, 1)
        .expect("a hash mismatch must not abort the read");
    let _ = trajectory
        .util_pos_read_range_f32(0, 0)
        .expect("a hash mismatch must not abort the read");
    assert_eq!(trajectory.hash_mismatch_count(), 1);
    assert_eq!(trajectory.mismatched_block_ids(), [BlockID::TrajPositions]);
}

#[test]
fn intact_trajectory_reports_no_hash_mismatch() {
    let spec = Spec::new();
    let (_directory, path) = write_trajectory("intact-hash", &spec);
    let mut trajectory = open_trajectory(&path);
    let _ = trajectory.util_pos_read_range_f32(0, 0).unwrap();
    assert_eq!(trajectory.hash_mismatch_count(), 0);
    assert!(trajectory.mismatched_block_ids().is_empty());
}

/// `Int` blocks decode through the generic (`f64`) interval reader.
///
/// That reader converts each stored value with `interval_bytes_to_f64`, not with
/// `NumericValue::decode`, so this pins the `Int` conversion of the value-level path. The typed
/// `f32` readers must keep rejecting the same block with their own message.
#[test]
fn int_blocks_decode_through_the_generic_reader() {
    let spec = Spec {
        positions_type: DataType::Int,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("int-block", &spec);
    let mut trajectory = open_trajectory(&path);

    let (values, n_frames, n_particles, n_values_per_frame, data_type) = trajectory
        .particle_data_interval_get(BlockID::TrajPositions, 0, 0, true)
        .expect("an Int block must decode through the generic reader");
    assert_eq!(n_frames, 1);
    assert_eq!(n_particles, PARTICLES);
    assert_eq!(n_values_per_frame, VALUES_PER_FRAME);
    assert_eq!(data_type, DataType::Int);

    // `sample_value` is fractional; the payload stores `value as i64`, so the expected values
    // truncate before being widened back to f64.
    let expected: Vec<f64> = expected_particle_values(&spec, 0.0, 0, 0)
        .into_iter()
        .map(|value| f64::from(value as i64 as f32))
        .collect();
    assert_eq!(values, expected);

    // The Float-only typed reader must reject an Int block rather than misreading it.
    let error = trajectory
        .util_pos_read_range_f32(0, 0)
        .expect_err("reading Int data as f32 must fail");
    assert!(
        matches!(&error, TngError::Constraint(message) if message.contains("as f32")),
        "unexpected error: {error:?}"
    );
}

/// `particle_data_get` decodes through `NumericValue::decode`, so it is the public entry point
/// that exercises the `Int` arm of the dispatch (`decode_i64`).
///
/// `particle_data_interval_get` uses `interval_bytes_to_f64` instead, so covering it does *not*
/// reach that arm. This pins the vector path, including the `f64` widening of stored `i64`
/// values, and the error a `f32` read of the same block must produce.
#[test]
fn int_blocks_decode_through_the_vector_reader() {
    let spec = Spec {
        positions_type: DataType::Int,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("int-vector", &spec);
    let mut trajectory = open_trajectory(&path);
    // `particle_data_get` reads from the last read frame set.
    trajectory
        .frame_set_read_next(true)
        .expect("read frame set");

    let (values, n_frames, n_particles, n_values_per_frame, data_type) = trajectory
        .particle_data_get(BlockID::TrajPositions)
        .expect("an Int block must decode through the vector reader");
    assert_eq!(n_frames, 1);
    assert_eq!(n_particles, PARTICLES);
    assert_eq!(n_values_per_frame, VALUES_PER_FRAME);
    assert_eq!(data_type, DataType::Int);

    // The payload stores `value as i64`, so the expected values truncate first.
    let expected: Vec<f64> = expected_particle_values(&spec, 0.0, 0, 0)
        .into_iter()
        .map(|value| f64::from(value as i64 as f32))
        .collect();
    assert_eq!(values, expected);
}

/// The block metadata getters must resolve blocks that were written into a frame set.
///
/// They have three lookup layers (non-trajectory vectors, then `data_block_ref`, then a
/// re-scan), and nothing covered them: `data_block_name_get`,
/// `data_block_dependency_get` and `data_block_num_values_per_frame_get` had no callers.
#[test]
fn block_metadata_getters_resolve_written_blocks() {
    let spec = Spec::new();
    let (_directory, path) = write_trajectory("metadata-getters", &spec);
    let mut trajectory = open_trajectory(&path);
    trajectory
        .frame_set_read_next(true)
        .expect("read frame set");

    assert_eq!(
        trajectory
            .data_block_name_get(BlockID::TrajPositions)
            .expect("positions block name"),
        "POSITIONS"
    );
    assert_eq!(
        trajectory
            .data_block_num_values_per_frame_get(BlockID::TrajPositions)
            .expect("positions values per frame"),
        VALUES_PER_FRAME
    );

    // Positions are particle dependent and frame dependent.
    let dependency = trajectory
        .data_block_dependency_get(BlockID::TrajPositions)
        .expect("positions dependency");
    assert_ne!(dependency, 0, "particle data must be particle dependent");

    // Box shape is written as a non-particle block, so it must resolve too.
    assert_eq!(
        trajectory
            .data_block_name_get(BlockID::TrajBoxShape)
            .expect("box shape block name"),
        "BOX SHAPE"
    );
    assert_eq!(
        trajectory
            .data_block_num_values_per_frame_get(BlockID::TrajBoxShape)
            .expect("box shape values per frame"),
        BOX_VALUES
    );
    assert_eq!(
        trajectory
            .data_block_dependency_get(BlockID::TrajBoxShape)
            .expect("box shape dependency"),
        1,
        "box shape is frame dependent but not particle dependent"
    );
}

/// Walk the TNG block headers and return the byte range of the contents of `id`.
///
/// The general block header is `header_contents_size` (`u64`, so it covers the whole header),
/// `block_contents_size` (`u64`), id (`u64`), version (`u64`), a 16-byte MD5 hash and the
/// length-prefixed block name; the payload starts at `offset + header_contents_size`. The file
/// endianness is inferred exactly like [`Trajectory`] does, so this works on big-endian hosts.
fn find_block_contents(bytes: &[u8], id: BlockID) -> Option<std::ops::Range<usize>> {
    let little_endian = detect_file_little_endian(bytes)?;
    let read_u64 = |at: usize| -> u64 {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&bytes[at..at + 8]);
        if little_endian {
            u64::from_le_bytes(raw)
        } else {
            u64::from_be_bytes(raw)
        }
    };

    let mut offset = 0usize;
    while offset + 32 <= bytes.len() {
        let header_contents_size = usize::try_from(read_u64(offset)).ok()?;
        let block_contents_size = usize::try_from(read_u64(offset + 8)).ok()?;
        let block_id = read_u64(offset + 16);

        if header_contents_size == 0 {
            break;
        }
        if block_id == id as u64 {
            let start = offset + header_contents_size;
            return Some(start..start + block_contents_size);
        }

        let next = (offset + header_contents_size).checked_add(block_contents_size)?;
        if next <= offset || next > bytes.len() {
            break;
        }
        offset = next;
    }
    None
}

/// Infer the TNG file endianness from the first block header, mirroring the parser.
///
/// A header whose most significant byte is zero and whose least significant byte is non-zero
/// is little endian; every other pattern is big endian (`Trajectory::block_header_read`). The
/// big-endian case includes small header sizes, whose most significant byte is legitimately
/// zero, so it must not be treated as "unknown".
fn detect_file_little_endian(bytes: &[u8]) -> Option<bool> {
    let header_contents_size = bytes.get(..8)?;
    Some(header_contents_size[0] != 0 && header_contents_size[7] == 0)
}

/// `find_block_contents` must locate a block in a little- *and* a big-endian file.
///
/// A real big-endian host is required to *write* such a file, so this builds the two-byte-order
/// encodings by hand: the layout is the same on every host, only the integer byte order differs,
/// and the parser reaches the same conclusion from the same first header.
#[test]
fn find_block_contents_handles_both_byte_orders() {
    const HEADER: u64 = 0x48;
    const CONTENTS: u64 = 0x10;

    let mut little = Vec::new();
    little.extend_from_slice(&HEADER.to_le_bytes());
    little.extend_from_slice(&CONTENTS.to_le_bytes());
    little.extend_from_slice(&(BlockID::TrajPositions as u64).to_le_bytes());
    little.resize(HEADER as usize, 0);
    little.extend_from_slice(&[0xAB; CONTENTS as usize]);

    let mut big = Vec::new();
    big.extend_from_slice(&HEADER.to_be_bytes());
    big.extend_from_slice(&CONTENTS.to_be_bytes());
    big.extend_from_slice(&(BlockID::TrajPositions as u64).to_be_bytes());
    big.resize(HEADER as usize, 0);
    big.extend_from_slice(&[0xCD; CONTENTS as usize]);

    let expected = HEADER as usize..(HEADER + CONTENTS) as usize;
    assert_eq!(
        find_block_contents(&little, BlockID::TrajPositions),
        Some(expected.clone())
    );
    assert_eq!(
        find_block_contents(&big, BlockID::TrajPositions),
        Some(expected)
    );

    // A header that is absent must not be reported as found.
    assert_eq!(find_block_contents(&big, BlockID::TrajVelocities), None);
}

#[test]
fn per_frame_loop_reads_every_block_from_its_own_frame_set() {
    // A per-frame loop reads several blocks of each frame set. Every block has to come from the
    // frame set that holds the frame, not from the frame set it was first loaded for.
    let spec = Spec {
        frames_per_set: 10,
        frame_sets: 3,
        ..Spec::new()
    };
    let (_directory, path) = write_trajectory("per-frame-loop", &spec);
    let mut trajectory = open_trajectory(&path);

    for frame in 0..spec.frames_per_set * spec.frame_sets {
        let (positions, _) = trajectory
            .util_pos_read_range_f32(frame, frame)
            .unwrap_or_else(|error| panic!("positions at frame {frame}: {error}"));
        assert_close_f32(
            &positions,
            &expected_particle_values(&spec, 0.0, frame, frame),
            0.0,
            &format!("positions at frame {frame}"),
        );

        let (velocities, _) = trajectory
            .util_vel_read_range_f32(frame, frame)
            .unwrap_or_else(|error| panic!("velocities at frame {frame}: {error}"));
        assert_close_f32(
            &velocities,
            &expected_particle_values(&spec, VELOCITY_OFFSET, frame, frame),
            0.0,
            &format!("velocities at frame {frame}"),
        );

        let (box_shape, _) = trajectory
            .util_box_shape_read_range_f32(frame, frame)
            .unwrap_or_else(|error| panic!("box shape at frame {frame}: {error}"));
        assert_close_f32(
            &box_shape,
            &expected_box_values(&spec, frame, frame),
            0.0,
            &format!("box shape at frame {frame}"),
        );
    }
}
