use assert_approx_eq::assert_approx_eq;
use tng_rs::trajectory::Trajectory;
const TEST_FILES_DIR: &str = "test_files";

fn molecule_find(traj: &mut Trajectory) {
    traj.molecule_find(Some("Argon"), None).unwrap();
}

fn num_particles(traj: &mut Trajectory) {
    let n_particles = traj.num_particles_get();
    assert_eq!(n_particles, 1000);
}

fn num_frames(traj: &mut Trajectory) {
    let n_frames = traj.num_frames_get().unwrap();
    assert_eq!(n_frames, 500001);
}

fn box_shape_read_values(traj: &mut Trajectory) {
    let (box_shape, _stride_length) = traj.util_box_shape_read().unwrap();
    let frame_0 = [
        3.60140, 0.00000, 0.000000, 0.000000, 3.60140, 0.000000, 0.000000, 0.000000, 3.60140,
    ];
    assert_eq!(box_shape[..9], frame_0);

    let frame_100 = [
        3.589650, 0.000000, 0.000000, 0.000000, 3.589650, 0.000000, 0.000000, 0.000000, 3.589650,
    ];
    for i in 0..9 {
        assert_approx_eq!(box_shape[909 - 9 + i], frame_100[i]);
    }
}
fn position_partial_read(traj: &mut Trajectory) {
    let (_positions, stride_length) = traj.util_pos_read_range(0, 5000).unwrap();
    assert_eq!(stride_length, 5000);
}
fn position_partial_read_invalid_range(traj: &mut Trajectory) {
    let _err = traj
        .util_pos_read_range(1, 1)
        .expect_err("should be invalid range");
}
fn position_partial_values_frm0(traj: &mut Trajectory) {
    let (positions, stride_length) = traj.util_pos_read_range(0, 0).unwrap();
    assert_eq!(stride_length, 5000);

    // xyz first 10 atoms frame 0
    // gmx dump frame 0
    #[rustfmt::skip]
    let frame_0_first_10_values = [
        2.53300e+00,  1.24400e+00,  3.50600e+00,
        8.30000e-01,  2.54400e+00,  3.44800e+00,
        1.09100e+00,  1.10000e-01,  3.12900e+00,
        2.45500e+00,  5.00000e-03,  3.01200e+00,
        2.71400e+00,  1.35300e+00,  5.53000e-01,
        3.05100e+00,  2.89300e+00,  2.69100e+00,
        1.42200e+00,  2.77000e+00,  1.46000e-01,
        2.22300e+00,  1.21100e+00,  3.26800e+00,
        2.81100e+00,  2.78900e+00,  2.38500e+00,
        4.87000e-01,  1.15900e+00,  1.17100e+00,
    ];

    for i in 0..30 {
        assert_approx_eq!(positions[i], frame_0_first_10_values[i]);
    }
}
fn position_partial_values_frm1(traj: &mut Trajectory) {
    let (positions, stride_length) = traj.util_pos_read_range(5000, 5000).unwrap();
    assert_eq!(stride_length, 5000);

    // xyz first 10 atoms frame 1
    // gmx dump frame 1
    #[rustfmt::skip]
    let frame_1_first_10_values = [
        2.52400e+00,  1.18600e+00,  2.33000e-01,
        9.01000e-01,  2.77300e+00,  3.13500e+00,
        1.68400e+00,  2.14000e-01,  3.17900e+00,
        2.19300e+00,  3.37400e+00,  2.90700e+00,
        2.89400e+00,  1.50600e+00,  4.46000e-01,
        2.79600e+00,  2.87600e+00,  2.54400e+00,
        1.37600e+00,  3.06700e+00,  1.78000e-01,
        2.17900e+00,  9.04000e-01,  3.04800e+00,
        2.68400e+00,  2.53800e+00,  2.47300e+00,
        3.78000e-01,  1.41000e+00,  8.46000e-01,
    ];
    for i in 0..30 {
        assert_approx_eq!(positions[i], frame_1_first_10_values[i]);
    }
}

fn position_partial_read_irregular(traj: &mut Trajectory) {
    let (_positions, stride_length) = traj.util_pos_read_range(7777, 18888).unwrap();
    assert_eq!(stride_length, 5000);
}

fn position_read(traj: &mut Trajectory) {
    let (_positions, stride_length) = traj.util_pos_read().unwrap();
    assert_eq!(stride_length, 5000);
}

fn position_values(traj: &mut Trajectory) {
    let (positions, _stride_length) = traj.util_pos_read().unwrap();
    // xyz first 10 atoms frame 0
    #[rustfmt::skip]
    let frame_0_first_10_values = [
        2.53300e+00,  1.24400e+00,  3.50600e+00,
        8.30000e-01,  2.54400e+00,  3.44800e+00,
        1.09100e+00,  1.10000e-01,  3.12900e+00,
        2.45500e+00,  5.00000e-03,  3.01200e+00,
        2.71400e+00,  1.35300e+00,  5.53000e-01,
        3.05100e+00,  2.89300e+00,  2.69100e+00,
        1.42200e+00,  2.77000e+00,  1.46000e-01,
        2.22300e+00,  1.21100e+00,  3.26800e+00,
        2.81100e+00,  2.78900e+00,  2.38500e+00,
        4.87000e-01,  1.15900e+00,  1.17100e+00,
    ];
    for i in 0..30 {
        assert_approx_eq!(positions[i], frame_0_first_10_values[i]);
    }

    // xyz last 10 atoms frame 100
    #[rustfmt::skip]
    let frame_100_last_10_values = [
        7.76000e-01,  1.19600e+00,  7.73000e-01,
        6.27000e-01,  3.34000e-01,  2.04900e+00,
        6.09000e-01,  3.46300e+00,  2.57000e-01,
        3.02000e+00,  3.18400e+00,  2.97600e+00,
        2.64700e+00,  7.74000e-01,  1.81500e+00,
        1.56000e-01,  1.28300e+00,  3.28100e+00,
        6.58000e-01,  3.03300e+00,  2.90800e+00,
        2.08500e+00,  3.55100e+00,  1.43600e+00,
        1.56000e-01,  3.50200e+00,  3.14000e-01,
        1.28900e+00,  9.98000e-01,  1.64500e+00,
    ];
    for i in 0..30 {
        assert_approx_eq!(positions[303000 - 30 + i], frame_100_last_10_values[i]);
    }
}
fn force_read(traj: &mut Trajectory) {
    let _ = traj
        .util_force_read()
        .expect_err("no forces expected in this file");
}
fn vel_read(traj: &mut Trajectory) {
    let _ = traj
        .util_vel_read()
        .expect_err("no velocities expected in this file");
}
fn num_molecule_types(traj: &mut Trajectory) {
    let count = traj.num_molecule_types_get();
    assert_eq!(count, 1);
}
fn num_molecules(traj: &mut Trajectory) {
    let count = traj.num_molecules_get();
    assert_eq!(count, 1000);
}

#[test]
fn argon_compressed() {
    let mut input_filename = std::env::current_dir().expect("able to get current working dir");
    input_filename.push(TEST_FILES_DIR);
    input_filename.push("argon_npt_compressed.tng");

    let mut traj = Trajectory::new();
    traj.util_trajectory_open(input_filename.as_path(), 'r')
        .unwrap();

    num_particles(&mut traj);
    num_frames(&mut traj);
    box_shape_read_values(&mut traj);
    position_partial_read(&mut traj);
    position_partial_read_invalid_range(&mut traj);
    position_partial_values_frm0(&mut traj);
    position_partial_values_frm1(&mut traj);
    position_partial_read_irregular(&mut traj);
    position_read(&mut traj);
    position_values(&mut traj);
    force_read(&mut traj);
    vel_read(&mut traj);
    num_molecule_types(&mut traj);
    num_molecules(&mut traj);
    molecule_find(&mut traj);
}

fn tng_example_num_particles(traj: &mut Trajectory) {
    let n_particles = traj.num_particles_get();
    assert_eq!(n_particles, 15);
}

fn tng_example_num_frames(traj: &mut Trajectory) {
    let n_frames = traj.num_frames_get().unwrap();
    assert_eq!(n_frames, 10);
}

fn tng_example_box_shape_read(traj: &mut Trajectory) {
    let _err = traj
        .util_box_shape_read()
        .expect_err("no box shape was expected");
}
fn tng_example_box_read_then_position_partial_read(traj: &mut Trajectory) {
    let _err = traj
        .util_box_shape_read()
        .expect_err("normal box not expected");
    let (_positions, stride_length) = traj.util_pos_read_range(0, 2).unwrap();
    assert_eq!(stride_length, 1);
}
fn tng_example_position_read(traj: &mut Trajectory) {
    let (_positions, stride_length) = traj.util_pos_read().unwrap();
    assert_eq!(stride_length, 1);
}
fn tng_example_position_values(traj: &mut Trajectory) {
    let (positions, _stride_length) = traj.util_pos_read().unwrap();

    // xyz first 10 atoms frame 0
    // gmx dump frame 0
    #[rustfmt::skip]
    let frame_0_first_10_values = [
        1.00000e+00,  1.00000e+00,  1.00000e+00,
        2.00000e+00,  2.00000e+00,  2.00000e+00,
        3.00000e+00,  3.00000e+00,  3.00000e+00,
        1.10000e+01,  1.10000e+01,  1.10000e+01,
        1.20000e+01,  1.20000e+01,  1.20000e+01,
        1.30000e+01,  1.30000e+01,  1.30000e+01,
        2.10000e+01,  2.10000e+01,  2.10000e+01,
        2.20000e+01,  2.20000e+01,  2.20000e+01,
        2.30000e+01,  2.30000e+01,  2.30000e+01,
        8.25000e+00,  3.30000e+01,  3.30000e+01,
    ];
    for i in 0..30 {
        assert_approx_eq!(positions[i], frame_0_first_10_values[i]);
    }

    // xyz first 10 atoms frame 9
    // gmx dump frame 9
    #[rustfmt::skip]
    let frame_9_last_10_values = [
        1.30000e+01,  1.30000e+01,  1.30000e+01,
        2.10000e+01,  2.10000e+01,  2.10000e+01,
        2.20000e+01,  2.20000e+01,  2.20000e+01,
        2.30000e+01,  2.30000e+01,  2.30000e+01,
        8.25000e+00,  3.30000e+01,  3.30000e+01,
        8.25000e+00,  3.40000e+01,  3.30000e+01,
        8.50000e+00,  3.30000e+01,  3.40000e+01,
        5.00000e+01,  5.00000e+01,  5.00000e+01,
        5.10000e+01,  5.10000e+01,  5.10000e+01,
        1.00000e+02,  1.00000e+02,  1.00000e+02,
    ];
    for i in 0..30 {
        assert_approx_eq!(positions[450 - 30 + i], frame_9_last_10_values[i]);
    }
}
fn tng_example_force_read(traj: &mut Trajectory) {
    let _ = traj
        .util_force_read()
        .expect_err("no forces expected in this file");
}
fn tng_example_vel_read(traj: &mut Trajectory) {
    let _ = traj
        .util_vel_read()
        .expect_err("no velocities expected in this file");
}
fn tng_example_num_molecule_types(traj: &mut Trajectory) {
    let count = traj.num_molecule_types_get();
    assert_eq!(count, 1);
}
fn tng_example_num_molecules(traj: &mut Trajectory) {
    let count = traj.num_molecules_get();
    assert_eq!(count, 5);
}
fn tng_example_molecule_find(traj: &mut Trajectory) {
    let _ = traj.molecule_find(Some("water"), None).unwrap();
}

#[test]
fn tng_example_test() {
    let mut input_filename = std::env::current_dir().expect("able to get current working dir");
    input_filename.push(TEST_FILES_DIR);
    input_filename.push("tng_example.tng");

    let mut traj = Trajectory::new();
    traj.util_trajectory_open(input_filename.as_path(), 'r')
        .unwrap();

    tng_example_num_particles(&mut traj);

    tng_example_num_frames(&mut traj);

    tng_example_box_shape_read(&mut traj);

    tng_example_box_read_then_position_partial_read(&mut traj);

    tng_example_position_read(&mut traj);

    tng_example_position_values(&mut traj);

    tng_example_force_read(&mut traj);

    tng_example_vel_read(&mut traj);

    tng_example_num_molecule_types(&mut traj);

    tng_example_num_molecules(&mut traj);

    tng_example_molecule_find(&mut traj);
}
