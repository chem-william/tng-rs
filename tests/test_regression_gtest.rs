use assert_approx_eq::assert_approx_eq;
use tng_rs::trajectory::Trajectory;
const TEST_FILES_DIR: &str = "test_files";

const TOL: f32 = 1e-5;
// larger as forces are larger
const FORCE_TOL: f32 = 1e-2;

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
        3.601_40, 0.00000, 0.000000, 0.000000, 3.601_40, 0.000000, 0.000000, 0.000000, 3.601_40,
    ];
    for i in 0..9 {
        assert_approx_eq!(box_shape[i], frame_0[i], TOL);
    }

    let frame_100 = [
        3.589_65, 0.000000, 0.000000, 0.000000, 3.589_65, 0.000000, 0.000000, 0.000000, 3.589_65,
    ];
    for i in 0..9 {
        assert_approx_eq!(box_shape[909 - 9 + i], frame_100[i], TOL);
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
        assert_approx_eq!(positions[i], frame_0_first_10_values[i], f64::from(TOL));
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
        assert_approx_eq!(positions[i], frame_1_first_10_values[i], f64::from(TOL));
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
        assert_approx_eq!(positions[i], frame_0_first_10_values[i], TOL);
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
        assert_approx_eq!(
            positions[303_000 - 30 + i],
            frame_100_last_10_values[i],
            TOL
        );
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
        assert_approx_eq!(positions[i], frame_0_first_10_values[i], TOL);
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
        assert_approx_eq!(positions[450 - 30 + i], frame_9_last_10_values[i], TOL);
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
fn tng_example() {
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

fn water_trj_num_particles(traj: &mut Trajectory) {
    let n_particles = traj.num_particles_get();
    assert_eq!(n_particles, 2700);
}
fn water_trj_num_frames(traj: &mut Trajectory) {
    let n_frames = traj.num_frames_get().unwrap();
    assert_eq!(n_frames, 500_001);
}
fn water_trj_box_shape_read(traj: &mut Trajectory) {
    let (_box_shape, stride_length) = traj.util_box_shape_read().unwrap();
    assert_eq!(stride_length, 5000);
}
fn water_trj_box_shape_values(traj: &mut Trajectory) {
    let (box_shape, _stride_length) = traj.util_box_shape_read().unwrap();
    let frame_0 = [
        3.01125e+00,
        0.00000e+00,
        0.00000e+00,
        0.00000e+00,
        3.01125e+00,
        0.00000e+00,
        0.00000e+00,
        0.00000e+00,
        3.01125e+00,
    ];
    for i in 0..9 {
        assert_approx_eq!(box_shape[i], frame_0[i], TOL);
    }

    let frame_100 = [
        2.870_21, 0.000000, 0.000000, 0.000000, 2.870_21, 0.000000, 0.000000, 0.000000, 2.870_21,
    ];
    for i in 0..9 {
        assert_approx_eq!(box_shape[909 - 9 + i], frame_100[i], TOL);
    }
}
fn water_trj_position_read(traj: &mut Trajectory) {
    let (_positions, stride_length) = traj.util_pos_read().unwrap();
    assert_eq!(stride_length, 5000);
}
fn water_trj_position_values(traj: &mut Trajectory) {
    let (positions, _stride_length) = traj.util_pos_read().unwrap();
    // xyz first 10 atoms frame 0
    // gmx dump frame 100
    #[rustfmt::skip]
    let frame_0_first_10_values = [
        7.43000e-01,  2.42200e+00,  2.25100e+00,
        7.29000e-01,  2.48600e+00,  2.32000e+00,
        6.60000e-01,  2.37500e+00,  2.24500e+00,
        9.44000e-01,  1.43200e+00,  1.51800e+00,
        1.02100e+00,  1.48600e+00,  1.50000e+00,
        8.76000e-01,  1.46800e+00,  1.46000e+00,
        2.55700e+00,  2.11600e+00,  1.38800e+00,
        2.64500e+00,  2.14600e+00,  1.41200e+00,
        2.50000e+00,  2.15500e+00,  1.45400e+00,
        1.04500e+00,  2.51300e+00,  2.47000e-01,
    ];
    for i in 0..30 {
        assert_approx_eq!(positions[i], frame_0_first_10_values[i], TOL);
    }

    // xyz last 10 atoms frame 100
    // gmx dump frame 100
    #[rustfmt::skip]
    let frame_100_last_10_values = [
        8.56000e-01,  2.24800e+00,  2.79100e+00,
        9.53000e-01,  7.58000e-01,  8.59000e-01,
        9.24000e-01,  7.24000e-01,  7.74000e-01,
        8.73000e-01,  7.67000e-01,  9.10000e-01,
        5.08000e-01,  2.31100e+00,  1.85000e-01,
        5.25000e-01,  2.32700e+00,  9.30000e-02,
        5.14000e-01,  2.21600e+00,  1.95000e-01,
        6.67000e-01,  2.15500e+00,  1.65700e+00,
        5.84000e-01,  2.14800e+00,  1.61000e+00,
        7.18000e-01,  2.21700e+00,  1.60500e+00,
    ];
    for i in 0..30 {
        assert_approx_eq!(
            positions[818_100 - 30 + i],
            frame_100_last_10_values[i],
            TOL
        );
    }
}
fn water_trj_force_read(traj: &mut Trajectory) {
    let _ = traj
        .util_force_read()
        .expect_err("no forces expected in this file");
}
fn water_trj_vel_read(traj: &mut Trajectory) {
    let _ = traj
        .util_vel_read()
        .expect_err("no forces expected in this file");
}
#[test]
fn water_trj_conv() {
    let mut input_filename = std::env::current_dir().expect("able to get current working dir");
    input_filename.push(TEST_FILES_DIR);
    input_filename.push("water_npt_compressed_trjconv.tng");

    let mut traj = Trajectory::new();
    traj.util_trajectory_open(input_filename.as_path(), 'r')
        .unwrap();

    water_trj_num_particles(&mut traj);

    water_trj_num_frames(&mut traj);

    water_trj_box_shape_read(&mut traj);

    water_trj_box_shape_values(&mut traj);

    water_trj_position_read(&mut traj);

    water_trj_position_values(&mut traj);

    water_trj_force_read(&mut traj);

    water_trj_vel_read(&mut traj);
}

fn water_vels_forces_box_shape_read(traj: &mut Trajectory) {
    let (_box_shape, stride_length) = traj.util_box_shape_read().unwrap();
    assert_eq!(stride_length, 5000);
}
fn water_vels_forces_box_shape_values(traj: &mut Trajectory) {
    let (box_shape, stride_length) = traj.util_box_shape_read().unwrap();
    assert_eq!(stride_length, 5000);

    let frame_0 = [
        2.87951e+00,
        0.00000e+00,
        0.00000e+00,
        0.00000e+00,
        2.87951e+00,
        0.00000e+00,
        0.00000e+00,
        0.00000e+00,
        2.87951e+00,
    ];
    for i in 0..9 {
        assert_approx_eq!(box_shape[i], frame_0[i], TOL);
    }

    let frame_100 = [
        2.89497e+00,
        0.00000e+00,
        0.00000e+00,
        0.00000e+00,
        2.89497e+00,
        0.00000e+00,
        0.00000e+00,
        0.00000e+00,
        2.89497e+00,
    ];
    for i in 0..9 {
        assert_approx_eq!(box_shape[909 - 9 + i], frame_100[i], TOL);
    }
}
fn water_vels_forces_position_read(traj: &mut Trajectory) {
    let (_positions, stride_length) = traj.util_pos_read().unwrap();
    assert_eq!(stride_length, 5000);
}
fn water_vels_forces_position_values(traj: &mut Trajectory) {
    let (positions, _stride_length) = traj.util_pos_read().unwrap();
    // xyz first 10 atoms frame 0
    // gmx dump frame 0
    #[rustfmt::skip]
    let frame_0_first_10_values = [
        2.52700e+00,  2.61101e+00,  2.45398e+00,
        2.50319e+00,  2.59390e+00,  2.54510e+00,
        2.61687e+00,  2.57898e+00,  2.44623e+00,
        1.09097e+00,  1.27301e+00,  1.99202e+00,
        1.01457e+00,  1.23310e+00,  2.03366e+00,
        1.13694e+00,  1.19976e+00,  1.95100e+00,
        2.20399e+00,  1.37297e+00,  8.83017e-01,
        2.13535e+00,  1.38523e+00,  9.48592e-01,
        2.21780e+00,  1.46022e+00,  8.46139e-01,
        1.10605e+00,  2.11799e+00,  5.61040e-01,
    ];
    for i in 0..30 {
        assert_approx_eq!(positions[i], frame_0_first_10_values[i], TOL);
    }

    // xyz last 10 atoms frame 100
    // gmx dump frame 100
    #[rustfmt::skip]
    let frame_100_last_10_values = [
        7.98970e-01,  2.15481e+00,  2.75854e+00,
        6.32804e-01,  6.59262e-01,  1.12701e+00,
        5.47739e-01,  6.89158e-01,  1.09488e+00,
        6.16521e-01,  5.70554e-01,  1.15907e+00,
        5.33961e-01,  2.20212e+00,  6.22357e-02,
        4.79836e-01,  2.17921e+00,  1.37788e-01,
        4.79169e-01,  2.18181e+00,  2.88140e+00,
        5.76261e-01,  1.85258e+00,  1.69974e+00,
        6.60233e-01,  1.87443e+00,  1.74016e+00,
        5.79366e-01,  1.75766e+00,  1.68776e+00,
    ];
    for i in 0..30 {
        assert_approx_eq!(
            positions[818_100 - 30 + i],
            frame_100_last_10_values[i],
            TOL
        );
    }
}
fn water_vels_forces_forces_read(traj: &mut Trajectory) {
    let (_forces, stride_length) = traj.util_force_read().unwrap();
    assert_eq!(stride_length, 5000);
}
fn water_vels_forces_forces_values(traj: &mut Trajectory) {
    let (forces, _stride_length) = traj.util_force_read().unwrap();
    // forces first 10 atoms frame 0
    #[rustfmt::skip]
    let frame_0_first_10_values = [
        -4.35261e+02,  3.36017e+02, -9.38570e+02,
        -1.75984e+01, -2.44064e+02,  1.25406e+03,
         6.57882e+02, -2.07715e+02,  2.72886e+02,
         1.75474e+01,  1.57273e+03,  2.80544e+01,
        -5.30602e+02, -8.79351e+02,  2.76766e+02,
         7.45154e+01, -5.15662e+02, -3.61260e+02,
         4.70405e+02, -1.26065e+03, -2.68651e+02,
        -5.15954e+02,  5.19739e+02,  2.85984e+02,
        -3.90010e+02,  4.82308e+02,  2.96046e+00,
         1.23199e+03, -7.51883e+02, -6.58181e+02,
    ];
    for i in 0..30 {
        assert_approx_eq!(forces[i], frame_0_first_10_values[i], FORCE_TOL);
    }
    // forces last 10 atoms frame 100
    #[rustfmt::skip]
    let frame_100_last_10_values = [
        -4.49360e+02, -5.46652e+02,  5.24477e+02,
         1.27648e+03,  8.27699e+02,  2.98916e+01,
        -9.49143e+02, -3.13201e+02, -3.78830e+02,
        -5.04814e+02, -5.57331e+02, -6.48604e+01,
         1.24046e+03,  1.05411e+03,  4.06005e+02,
        -3.61442e+02, -5.29395e+02,  1.26982e+02,
        -4.76165e+02, -5.24370e+02, -3.48132e+02,
        -7.41153e+02,  1.19924e+01, -7.19316e+02,
         5.67011e+02,  6.64948e+01,  2.13465e+02,
         2.43871e+02, -4.09309e+02,  4.87609e+01,
    ];
    for i in 0..30 {
        assert_approx_eq!(
            forces[818_100 - 30 + i],
            frame_100_last_10_values[i],
            FORCE_TOL
        );
    }
}
fn water_vels_forces_vel_read(traj: &mut Trajectory) {
    let (_velocities, stride_length) = traj.util_vel_read().unwrap();
    assert_eq!(stride_length, 5000);
}
fn water_vels_forces_vel_values(traj: &mut Trajectory) {
    let (velocities, _stride_length) = traj.util_vel_read().unwrap();
    // vels first 10 atoms frame 0
    #[rustfmt::skip]
    let frame_0_first_10_values = [
         3.51496e-01,  7.29674e-01, -5.33343e-02,
         5.97873e-02, -1.00359e+00, -4.19582e-01,
         2.56209e-01,  5.52850e-01, -4.53435e-01,
        -1.09184e-02,  3.66412e-01, -4.85018e-01,
         9.26847e-01, -6.03737e-01,  3.67032e-01,
        -9.85010e-02,  1.09447e+00, -1.94833e+00,
        -4.60571e-02,  3.64507e-01, -2.01200e-01,
        -1.23912e+00, -3.46699e-01, -1.27041e+00,
         6.12738e-01,  7.64292e-01,  9.39986e-01,
        -6.34257e-02, -3.96772e-02, -4.55601e-01,
    ];
    for i in 0..30 {
        assert_approx_eq!(velocities[i], frame_0_first_10_values[i], TOL);
    }
    // vels last 10 atoms frame 100
    #[rustfmt::skip]
    let frame_100_last_10_values = [
        -1.29712e+00,  1.89736e-01, -4.58020e-01,
        -2.24550e-01,  1.98991e-01, -7.18228e-01,
         9.92350e-02,  1.55654e-01, -1.64584e+00,
        -6.58128e-01,  4.26997e-01, -2.94439e-01,
        -2.47945e-01, -4.03298e-01,  2.42530e-01,
         3.88940e-01,  2.55276e-01,  9.15576e-01,
        -1.57709e+00,  5.61387e-01,  9.03308e-01,
        -5.50578e-01, -3.38237e-01, -9.82961e-02,
         4.52938e-01, -7.97070e-01, -1.83071e+00,
        -7.36810e-01, -2.02619e-01, -1.35719e+00,
    ];
    for i in 0..30 {
        assert_approx_eq!(
            velocities[818_100 - 30 + i],
            frame_100_last_10_values[i],
            TOL
        );
    }
}
#[test]
fn water_vels_forces() {
    let mut input_filename = std::env::current_dir().expect("able to get current working dir");
    input_filename.push(TEST_FILES_DIR);
    input_filename.push("water_uncompressed_vels_forces.tng");

    let mut traj = Trajectory::new();
    traj.util_trajectory_open(input_filename.as_path(), 'r')
        .unwrap();

    water_vels_forces_box_shape_read(&mut traj);
    water_vels_forces_box_shape_values(&mut traj);

    water_vels_forces_position_read(&mut traj);
    water_vels_forces_position_values(&mut traj);

    water_vels_forces_forces_read(&mut traj);
    water_vels_forces_forces_values(&mut traj);

    water_vels_forces_vel_read(&mut traj);
    water_vels_forces_vel_values(&mut traj);
}
