#![allow(dead_code)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TngError {
    /// A constraint or validation was violated (e.g. stride length ordering).
    /// Corresponds to C's `TNG_FAILURE` for argument/state validation.
    #[error("{0}")]
    Constraint(String),

    /// An item was not found (e.g. molecule not in trajectory).
    #[error("{0}")]
    NotFound(String),

    /// A major, unspecified error has occured - matches the C code.
    #[error("{0}")]
    Critical(String),

    /// I/O error wrapping `std::io::Error`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
mod atom;
mod bond;
mod bwlzh;
mod chain;
mod coder;
mod compress;
mod data;
mod dict;
mod fix_point;
mod gen_block;
mod huffman;
mod huffmem;
mod lz77;
mod molecule;
mod mtf;
mod particle_mapping;
mod residue;
mod rle;
pub mod trajectory;
mod trajectory_frame_set;
mod utils;
mod widemuldiv;
mod xtc2;
mod xtc3;

#[cfg(test)]
mod c_comparison;
#[cfg(test)]
mod ffi;

/// The maximum length of a date string
const MAX_DATE_STR_LEN: u64 = 24;
/// The length of an MD5 hash
const MD5_HASH_LEN: usize = 16;
/// The maximum allowed length of a string
const MAX_STR_LEN: usize = 1024;

const API_VERSION: u64 = 8;
/// Flag to indicate frame dependent data
const FRAME_DEPENDENT: u8 = 1;
/// Flag to indicate particle dependent data
const PARTICLE_DEPENDENT: u8 = 2;

#[cfg(test)]
mod integration {
    use crate::{
        FRAME_DEPENDENT, MAX_STR_LEN, PARTICLE_DEPENDENT,
        data::{Compression, DataType},
        gen_block::BlockID,
        molecule::Molecule,
        trajectory::{BlockType, Trajectory},
    };
    use assert_approx_eq::assert_approx_eq;
    use rand::RngExt;

    const TIME_PER_FRAME: f64 = 2e-15;
    const TEST_FILES_DIR: &str = "test_files";

    const BOX_SHAPE_X: f64 = 150.0;
    const BOX_SHAPE_Y: f64 = 145.5;
    const BOX_SHAPE_Z: f64 = 155.5;
    const N_FRAME_SETS: i64 = 100;
    const MEDIUM_STRIDE_LEN: i64 = 5;
    const LONG_STRIDE_LEN: i64 = 25;
    const USER_NAME: &str = "USER 1";
    const PROGRAM_NAME: &str = "tng_testing";
    const COMPUTER_NAME: &str = "Unknown computer";
    const FORCEFIELD_NAME: &str = "No forcefield";
    const COMPRESSION_PRECISION: f64 = 1000.0;
    const DISTANCE_UNIT_EXPONENTIAL: i64 = 9;

    const USE_HASH: bool = true;

    /// C API: tng_test_setup_molecules() in tng_io_testing.c:45
    fn setup_molecules(traj: &mut Trajectory) {
        let mol_idx = traj.add_molecule("water");
        let chain_idx = traj.add_chain(mol_idx, "W");
        let residue_idx = traj.chain_residue_add(mol_idx, chain_idx, "WAT");
        let _o_idx = traj.residue_atom_add(mol_idx, residue_idx, "O", "O");
        let _h1_idx = traj.residue_atom_add(mol_idx, residue_idx, "HO1", "H");
        let _h2_idx = traj.residue_atom_add(mol_idx, residue_idx, "HO2", "H");
        let _bond_idx = traj.add_molecule_bond(mol_idx, 0, 1);
        let _bond_idx = traj.add_molecule_bond(mol_idx, 0, 2);

        traj.molecule_cnt_set(mol_idx, 200);
        let count = traj.get_molecule_cnt(mol_idx);
        assert_eq!(count, 200);
    }

    /// C API: tng_test_molecules() in tng_io_testing.c:141
    fn test_molecules(traj: &mut Trajectory) {
        let cnt = traj.num_molecule_types_get();
        assert_eq!(cnt, 1, "Molecule reading error");

        let cnt = traj.num_molecules_get();
        assert_eq!(cnt, 200, "Molecule reading error");

        let var_atoms = traj.num_particles_variable_get();
        assert!(!var_atoms, "Molecule reading error");

        let molecule = traj.molecule_of_index_get(0).unwrap();
        traj.molecule_find(Some("water"), None).unwrap();

        traj.molecule_name_get(molecule, MAX_STR_LEN).unwrap();

        let cnt = traj.molecule_num_chains_get(molecule);
        assert_eq!(cnt, 1, "Cannot get number of chains in molecule.");

        let chain = traj.molecule_chain_of_index_get(molecule, 0).unwrap();

        molecule
            .chain_find("W", -1)
            .expect("'W' chain to be present");

        let cnt = traj.molecule_num_residues_get(molecule);
        assert_eq!(cnt, 1, "Cannot get number of residues in molecule.");

        traj.molecule_residue_of_index_get(molecule, 0).unwrap();

        let cnt = traj.molecule_num_atoms_get(molecule);
        assert_eq!(cnt, 3, "Cannot get number of atoms in molecule.");

        let atom = traj.molecule_atom_of_index_get(molecule, 0).unwrap();

        molecule.atom_find("O", -1).expect("'O' to be present");

        traj.chain_name_get(chain, MAX_STR_LEN).unwrap();

        let cnt = traj.chain_num_residues_get(chain);
        assert_eq!(cnt, 1, "Cannot get number of residues in chain.");

        let residue = traj.chain_residue_of_index_get(chain, 0).unwrap();

        traj.chain_residue_find(chain, Some("WAT"), None).unwrap();

        traj.residue_name_get(residue, MAX_STR_LEN).unwrap();

        let cnt = traj.residue_num_atoms_get(residue);
        assert_eq!(cnt, 3, "Cannot get number of atoms in residue.");

        traj.residue_atom_of_index_get(residue, 0).unwrap();

        traj.atom_name_get(atom, MAX_STR_LEN).unwrap();

        traj.atom_type_get(atom, MAX_STR_LEN).unwrap();

        let cnt = traj.molecule_id_of_particle_nr_get(0).unwrap();
        assert_eq!(cnt, 1, "Cannot get molecule id of atom");

        let cnt = traj.residue_id_of_particle_nr_get(0).unwrap();
        assert_eq!(cnt, 0, "Cannot get residue id of atom");

        let cnt = traj.global_residue_id_of_particle_nr_get(599).unwrap();
        assert_eq!(cnt, 199, "Cannot get global residue id of atom");

        traj.molecule_name_of_particle_nr_get(0, MAX_STR_LEN)
            .unwrap();

        traj.chain_name_of_particle_nr_get(0, MAX_STR_LEN).unwrap();

        traj.residue_name_of_particle_nr_get(0, MAX_STR_LEN)
            .unwrap();

        traj.atom_name_of_particle_nr_get(0, MAX_STR_LEN).unwrap();

        // tng_molecule_alloc + tng_molecule_name_set + tng_molecule_existing_add
        let mut molecule = Molecule::new();
        molecule.name = "TEST".to_string();
        traj.molecule_existing_add(molecule);

        // tng_molsystem_bonds_get
        let (bonds, from_atoms, to_atoms) =
            traj.molsystem_bonds_get().expect("molsystem to have bonds");
        assert_eq!(bonds, 400);
        assert_eq!(from_atoms.len(), 400);
        assert_eq!(to_atoms.len(), 400);
    }

    /// C API: tng_test_read_and_write_file() in tng_io_testing.c:371
    fn test_read_and_write_file(traj: &mut Trajectory) {
        traj.file_headers_read(USE_HASH).unwrap();
        traj.file_headers_write(USE_HASH).unwrap();

        while traj.frame_set_read_next(USE_HASH).is_ok() {
            traj.frame_set_write(USE_HASH).unwrap();
        }
    }

    /// C API: tng_test_get_box_data() in tng_io_testing.c:926
    fn test_get_box_data(traj: &mut Trajectory) {
        let (box_data, _n_frames, _n_vpf, _dtype) = traj
            .data_get(BlockID::TrajBoxShape)
            .expect("Failed getting box shape");
        // The X dimension in the example file is 50
        assert!((box_data[0] - 50.0).abs() < 0.000001);
    }

    /// C API: tng_test_write_and_read_traj() in tng_io_testing.c:420
    fn write_and_read_traj(traj: &mut Trajectory, hash_mode: bool) {
        traj.set_medium_stride_length(MEDIUM_STRIDE_LEN).unwrap();
        traj.set_long_stride_length(LONG_STRIDE_LEN).unwrap();

        traj.first_user_name_set(USER_NAME);
        traj.set_first_program_name(PROGRAM_NAME);
        traj.first_computer_name_set(COMPUTER_NAME);
        traj.set_forcefield_name(FORCEFIELD_NAME);

        traj.compression_precision = COMPRESSION_PRECISION;
        traj.distance_unit_exponential = DISTANCE_UNIT_EXPONENTIAL;
        traj.set_time_per_frame(TIME_PER_FRAME).unwrap();

        // Create molecules
        setup_molecules(traj);

        // Set the box shape
        let mut box_shape = [0.0; 9];
        box_shape[0] = BOX_SHAPE_X;
        box_shape[4] = BOX_SHAPE_Y;
        box_shape[8] = BOX_SHAPE_Z;
        let bytes: Vec<_> = box_shape.iter().flat_map(|f| f.to_ne_bytes()).collect();
        traj.data_block_add(
            BlockID::TrajBoxShape,
            "BOX SHAPE",
            DataType::Double,
            &BlockType::NonTrajectory,
            1,
            9,
            1,
            Compression::Uncompressed,
            Some(&bytes),
        )
        .unwrap();

        // Set the partial charges (treat the water as TIP3P)
        let n_particles = traj.num_particles_get();
        let mut charges = vec![0.0_f32; usize::try_from(n_particles).expect("i64 to usize")];
        for i in 0..n_particles {
            let atom_type = traj.atom_type_of_particle_nr_get(i);

            // We only have water in the system. If the atom is oxygen set its
            // partial charge to -0.834, if it's a hydrogen set its partial charge to
            // 0.417
            match atom_type.chars().next().unwrap() {
                'O' => charges[i as usize] = -0.834,
                'H' => charges[i as usize] = 0.417,
                _ => unreachable!("failed to set partial charges"),
            }
        }

        let charges_bytes: Vec<_> = charges
            .iter()
            .flat_map(|&f: &f32| f.to_ne_bytes())
            .collect();
        traj.particle_data_block_add(
            BlockID::TrajPartialCharges,
            "PARTIAL CHARGES",
            DataType::Float,
            &BlockType::NonTrajectory,
            1,
            1,
            1,
            0,
            n_particles,
            Compression::Uncompressed,
            Some(&charges_bytes),
        )
        .unwrap();

        // Set atom masses
        let mut masses = vec![0.0_f32; usize::try_from(n_particles).expect("i64 to usize")];
        for i in 0..n_particles {
            let atom_type = traj.atom_type_of_particle_nr_get(i);
            // We only have water in the system. If the atom is oxygen set its
            // mass to 16.00000, if it's a hydrogen set its mass to
            // 1.00800.
            match atom_type.chars().next().unwrap() {
                'O' => masses[i as usize] = 16.00000,
                'H' => masses[i as usize] = 1.008,
                _ => unreachable!("failed to set atom masses"),
            }
        }

        let masses_bytes: Vec<_> = masses.iter().flat_map(|&f: &f32| f.to_ne_bytes()).collect();
        traj.particle_data_block_add(
            BlockID::TrajMasses,
            "ATOM MASSES",
            DataType::Float,
            &BlockType::NonTrajectory,
            1,
            1,
            1,
            0,
            n_particles,
            Compression::GZip,
            Some(&masses_bytes),
        )
        .unwrap();

        // Create a custom annotation data block
        let annotation =
            "This trajectory was generated from tng_io_testing. It is not a real MD trajectory.";
        traj.data_block_add(
            BlockID::TrajGeneralComments,
            "COMMENTS",
            DataType::Char,
            &BlockType::NonTrajectory,
            1,
            1,
            1,
            Compression::Uncompressed,
            Some(annotation.as_bytes()),
        )
        .expect("Failed adding details annotation data block");

        // Write file headers (includes non trajectory data blocks)
        traj.file_headers_write(USE_HASH).unwrap();

        let n_frames_per_frame_set = traj.num_frames_per_frame_set_get();
        let mut data = Vec::with_capacity(
            usize::try_from(n_particles * n_frames_per_frame_set * 3).expect("i64 to usize"),
        );

        let tot_n_mols = traj.get_num_molecules();
        // Set initial coordinates
        let mut rng = rand::rng();
        let mut molpos = vec![0.0_f32; tot_n_mols * 3];
        for i in 0..tot_n_mols {
            let nr = i * 3;
            // Somewhat random coordinates (between 0 and 100),
            // but not specifiying a random seed.
            molpos[nr] = 100.0 * rng.random_range(0.0_f32..1.0_f32);
            molpos[nr + 1] = 100.0 * rng.random_range(0.0_f32..1.0_f32);
            molpos[nr + 2] = 100.0 * rng.random_range(0.0_f32..1.0_f32);
        }

        // Generate frame sets - each with 100 frames (by default)
        for i in 0..N_FRAME_SETS {
            data.clear();
            let codec_id = if i < N_FRAME_SETS / 2 {
                Compression::GZip
            } else {
                Compression::TNG
            };

            for _j in 0..n_frames_per_frame_set {
                for k in 0..tot_n_mols {
                    let nr = k * 3;
                    // Move -1 to 1
                    molpos[nr] += 2.0 * rng.random_range(0.0_f32..1.0_f32) - 1.0;
                    molpos[nr + 1] += 2.0 * rng.random_range(0.0_f32..1.0_f32) - 1.0;
                    molpos[nr + 2] += 2.0 * rng.random_range(0.0_f32..1.0_f32) - 1.0;

                    data.push(molpos[nr]);
                    data.push(molpos[nr + 1]);
                    data.push(molpos[nr + 2]);

                    data.push(molpos[nr] + 1.0);
                    data.push(molpos[nr + 1] + 1.0);
                    data.push(molpos[nr + 2] + 1.0);

                    data.push(molpos[nr] - 1.0);
                    data.push(molpos[nr + 1] - 1.0);
                    data.push(molpos[nr + 2] - 1.0);
                }
            }
            traj.frame_set_with_time_new(
                i * n_frames_per_frame_set,
                n_frames_per_frame_set,
                2e-15f64 * (i * n_frames_per_frame_set) as f64,
            )
            .expect("error creating frame set");

            traj.frame_set_particle_mapping_free();

            // Setup particle mapping. Use 4 different mapping blocks with arbitrary
            // mappings.
            // C code rebuilds mapping array from scratch for each block
            let mapping: Vec<i64> = (0..150).collect();
            traj.particle_mapping_add(0, 150, &mapping).unwrap();

            let mapping: Vec<i64> = (0..150).map(|k| 599 - k).collect();
            traj.particle_mapping_add(150, 150, &mapping).unwrap();

            let mapping: Vec<i64> = (0..150).map(|k| k + 150).collect();
            traj.particle_mapping_add(300, 150, &mapping).unwrap();

            let mapping: Vec<i64> = (0..150).map(|k| 449 - k).collect();
            traj.particle_mapping_add(450, 150, &mapping).unwrap();

            // Add the positions in a data block
            let data_bytes: Vec<_> = data.iter().flat_map(|&f| f.to_ne_bytes()).collect();
            traj.particle_data_block_add(
                BlockID::TrajPositions,
                "POSITIONS",
                DataType::Float,
                &BlockType::Trajectory,
                n_frames_per_frame_set,
                3,
                1,
                0,
                n_particles,
                codec_id,
                Some(&data_bytes),
            )
            .unwrap();

            traj.frame_set_write(USE_HASH)
                .expect("error writing frame set");
        }

        // tng_io_testing.c:709-922
        *traj = Trajectory::new();
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        input_filename.push(TEST_FILES_DIR);
        input_filename.push("tng_test.tng");
        traj.input_file_set(input_filename.as_path());

        traj.file_headers_read(USE_HASH).unwrap();

        let temp_str = traj.first_user_name_get(MAX_STR_LEN).unwrap();
        assert_eq!(
            USER_NAME, temp_str,
            "User name does not match when reading written file"
        );

        let temp_str = traj.first_program_name_get(MAX_STR_LEN).unwrap();
        assert_eq!(
            PROGRAM_NAME, temp_str,
            "Program name does not match when reading written file"
        );
        let temp_str = traj.first_computer_name_get(MAX_STR_LEN).unwrap();
        assert_eq!(
            COMPUTER_NAME, temp_str,
            "Computer name does not match when reading written file"
        );

        let temp_str = traj.forcefield_name_get(MAX_STR_LEN).unwrap();
        assert_eq!(
            FORCEFIELD_NAME, temp_str,
            "Forcefield name does not match when reading written file"
        );

        let temp_int = traj.medium_stride_length_get();
        assert_eq!(
            MEDIUM_STRIDE_LEN, temp_int,
            "Stride length does not match when reading written file"
        );

        let temp_int = traj.long_stride_length_get();
        assert_eq!(
            LONG_STRIDE_LEN, temp_int,
            "Stride length does not match when reading written file"
        );

        let temp_double = traj.compression_precision_get();
        assert_eq!(
            COMPRESSION_PRECISION, temp_double,
            "Compression precision does not match when reading written file"
        );

        let temp_int = traj.distance_unit_exponential_get();
        assert_eq!(
            DISTANCE_UNIT_EXPONENTIAL, temp_int,
            "Distance unit exponential does not match when reading written file"
        );

        test_molecules(traj);

        let (masses, _n_frames, read_n_particles, _n_values_per_frame, _data_type) = traj
            .particle_data_get(BlockID::TrajMasses)
            .expect("failed getting particle masses");

        assert_eq!(
            read_n_particles, n_particles,
            "Number of particles does not match when reading atom masses."
        );
        // Above we have written only water molecules (in the order oxygen, hydrogen, hydrogen ...).
        // Test that the first and second as well as the very last atoms (oxygen, hydrogen and hydrogen)
        // have the correct atom masses.
        assert_approx_eq!((masses[0] - 16.0).abs(), 0.0);
        assert_approx_eq!((masses[1] - 1.008).abs(), 0.0);
        assert_approx_eq!((*masses.last().unwrap() - 1.008).abs(), 0.0);

        // Read all frame sets (tng_io_testing.c:804-842)
        let mut i = 0;
        while traj.frame_set_read_next(USE_HASH).is_ok() {
            let frame_set = traj.current_frame_set_get();
            let temp_int = traj.frame_set_prev_frame_set_file_pos(frame_set);
            let temp_int2 = traj.frame_set_next_frame_set_file_pos_get(frame_set);

            if i > 0 {
                assert_ne!(
                    temp_int, -1,
                    "File position of previous frame set not correct."
                );
            } else {
                assert_eq!(
                    temp_int, -1,
                    "File position of previous frame set not correct."
                );
            }
            if i < N_FRAME_SETS - 1 {
                assert_ne!(
                    temp_int2, -1,
                    "File position of next frame set not correct."
                );
            } else {
                assert_eq!(
                    temp_int2, -1,
                    "File position of previous next set not correct."
                );
            }
            i += 1;
        }

        let temp_double = traj.time_per_frame_get();
        assert_approx_eq!(temp_double, TIME_PER_FRAME, 1e-6);

        traj.frame_set_nr_find((0.3 * N_FRAME_SETS as f64) as i64)
            .expect(&format!(
                "Could not find frame set {}",
                (0.3 * N_FRAME_SETS as f64) as i64
            ));

        traj.frame_set_nr_find((0.75 * N_FRAME_SETS as f64) as i64)
            .expect(&format!(
                "Could not find frame set {}",
                (0.75 * N_FRAME_SETS as f64) as i64
            ));

        let frame_set = traj.current_frame_set_get();
        let (temp_int, _temp_int2) = traj.frame_set_frame_range_get(frame_set);
        assert_eq!(temp_int, 75 * n_frames_per_frame_set);

        traj.frame_set_read_current_only_data_from_block_id(hash_mode, BlockID::TrajPositions)
            .expect("Cannot read positions in current frame set.");

        traj.frame_set_read_next_only_data_from_block_id(hash_mode, BlockID::TrajPositions)
            .expect("Cannot read positions in next frame set.");
        let temp_str = traj
            .data_block_name_get(BlockID::TrajPositions)
            .expect("Could not get name of data block");
        assert_eq!("POSITIONS", temp_str, "Unexpected block name");

        traj.data_block_name_get(BlockID::TrajForces).expect_err(
            "Trying to retrieve name of non-existent data block did not return failure. %s: ",
        );
        let dependency = traj
            .data_block_dependency_get(BlockID::TrajPositions)
            .expect("Cannot get dependency of data block");
        assert_eq!(
            FRAME_DEPENDENT + PARTICLE_DEPENDENT,
            dependency,
            "Unexpected dependency"
        );

        let temp_int = traj
            .data_block_num_values_per_frame_get(BlockID::TrajPositions)
            .unwrap();
        assert_eq!(
            3, temp_int,
            "Cannot get number of values per frame of data block or unexpected value"
        );

        let temp_int = traj
            .data_get_stride_length(BlockID::TrajPositions, 100)
            .expect("Cannot get stride length of data block");
        assert_eq!(1, temp_int, "Unexpected value");
    }

    /// C API: tng_test_get_positions_data() in tng_io_testing.c:953
    /// This test relies on knowing that the positions are stored as float
    /// and that the data is not sparse (i.e. as many frames in the data as in the frame set)
    fn get_positions_data(traj: &mut Trajectory, hash_mode: bool) {
        let (values, n_frames, n_particles, n_values_per_frame, _data_type) = traj
            .particle_data_get(BlockID::TrajPositions)
            .expect("failed getting particle positions");
        assert_eq!(
            n_values_per_frame, 3,
            "Number of values per frame does not match expected value."
        );
        let n_frames = n_frames as usize;
        let n_particles = n_particles as usize;
        let n_values_per_frame = n_values_per_frame as usize;
        for i in 0..n_frames {
            for j in 0..n_particles {
                for k in 0..n_values_per_frame {
                    let value = values[(i * n_particles + j) * n_values_per_frame + k];
                    assert!(
                        (-500.0..=500.0).contains(&value),
                        "Coordinates not in range at frame {i}, particle {j}, component {k}: {value}"
                    );
                }
            }
        }

        assert!(
            traj.particle_data_interval_get(BlockID::TrajPositions, 111000, 111499, hash_mode)
                .is_err()
        );

        let (values, _n_frames, n_particles, n_values_per_frame, _data_type) = traj
            .particle_data_interval_get(BlockID::TrajPositions, 1000, 1050, hash_mode)
            .unwrap();

        for i in 0..50 {
            for j in 0..n_particles {
                for k in 0..n_values_per_frame {
                    let value = values[((i * n_particles + j) * n_values_per_frame + k) as usize];
                    assert!(
                        (-500.0..=500.0).contains(&value),
                        "Coordinates not in range at frame {i}, particle {j}, component {k}: {value}"
                    );
                }
            }
        }
    }

    /// C API: tng_test_utility_functions() in tng_io_testing.c:1036
    fn test_utility_functions(traj: &mut Trajectory, _hash_mode: bool) {
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        input_filename.push(TEST_FILES_DIR);
        input_filename.push("tng_test.tng");
        traj.util_trajectory_open(input_filename.as_path(), 'r')
            .unwrap();

        let time = traj.util_time_of_frame_get(50).unwrap();
        assert!(
            (time - 100e-13).abs() <= 0.000001,
            "Unexpected time at frame 50. Value: {time}, expected value: 100e-13"
        );

        let time = traj.util_time_of_frame_get(100).unwrap();
        assert!(
            (time - 200e-13).abs() <= 0.000001,
            "Unexpected time at frame 100. Value: {time}, expected value: 200e-13"
        );

        let n_frames_per_frame_set = traj.num_frames_per_frame_set_get();
        let n_frames = traj
            .util_num_frames_with_data_of_block_id_get(BlockID::TrajPositions)
            .unwrap();
        assert_eq!(
            n_frames,
            n_frames_per_frame_set * N_FRAME_SETS,
            "Unexpected number of frames with positions data. Value: {n_frames}, expected value: {}",
            n_frames_per_frame_set * N_FRAME_SETS
        );

        let n_frames_per_frame_set = traj.num_frames_per_frame_set_get();
        let n_frames = traj
            .util_num_frames_with_data_of_block_id_get(BlockID::TrajPositions)
            .unwrap();
        assert_eq!(
            n_frames,
            n_frames_per_frame_set * N_FRAME_SETS,
            "Unexpected number of frames with positions data. Value: {n_frames}, expected value: {}",
            n_frames_per_frame_set * N_FRAME_SETS
        );

        let n_particles = traj.num_particles_get();

        let n_frames_to_read = 30;
        let (positions, stride_length) = traj.util_pos_read_range(1, n_frames_to_read).unwrap();

        for i in 0..(n_frames_to_read / stride_length) as usize {
            for j in 0..n_particles as usize {
                for k in 0..3 {
                    let position = positions[i * n_particles as usize + j * 3 + k];
                    assert!(
                        (-500.0..=500.0).contains(&position),
                        "Coordinates not in range at frame {i}, particle {j}, component {k}: {position}"
                    );
                }
            }
        }

        let (next_frame, n_blocks) = traj
            .util_trajectory_next_frame_present_data_blocks_find(n_frames_to_read, 0, &[])
            .unwrap();
        assert_eq!(n_blocks, 1, "Unexpected data blocks in next frame.");
        assert_eq!(
            next_frame,
            n_frames_to_read + stride_length,
            "Unexpected data blocks in next frame."
        );

        let (codec_id, _factor) = traj
            .util_frame_current_compression_get(BlockID::TrajPositions)
            .unwrap();
        assert_eq!(codec_id, Compression::GZip, "Could not get compression");

        traj.util_trajectory_close().unwrap();
    }

    /// C API: tng_test_append() in tng_io_testing.c:1143
    fn test_append(traj: &mut Trajectory, hash_mode: bool) {
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        input_filename.push(TEST_FILES_DIR);
        input_filename.push("tng_test.tng");
        traj.util_trajectory_open(input_filename.as_path(), 'a')
            .unwrap();

        traj.last_user_name_set(USER_NAME);
        traj.last_user_name_get(MAX_STR_LEN).unwrap();

        traj.last_program_name_set(PROGRAM_NAME);
        traj.last_program_name_get(MAX_STR_LEN).unwrap();

        traj.last_computer_name_set(COMPUTER_NAME);
        traj.last_computer_name_get(MAX_STR_LEN).unwrap();

        traj.file_headers_write(hash_mode).unwrap();

        let n_frames = traj.num_frames_get().unwrap();
        traj.frame_set_of_frame_find(n_frames - 1).unwrap();
        let mut time = traj.util_time_of_frame_get(n_frames - 1).unwrap();
        time += TIME_PER_FRAME;
        let n_particles = traj.num_particles_get();

        let mut velocities = (0..n_particles as usize * 3)
            .map(|x| x as f64)
            .collect::<Vec<_>>();

        traj.util_vel_with_time_double_write(n_frames, time, &mut velocities)
            .unwrap();

        traj.util_trajectory_close().unwrap();
    }

    /// C API: tng_test_copy_container() in tng_io_testing.c:1228
    fn test_copy_container(traj: &mut Trajectory, _hash_mode: bool) {
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        input_filename.push(TEST_FILES_DIR);
        input_filename.push("tng_test.tng");
        traj.util_trajectory_open(input_filename.as_path(), 'a')
            .unwrap();

        let mut dest = traj.trajectory_init_from_src();

        traj.molecule_system_copy(&mut dest);

        traj.util_trajectory_close().unwrap();
        dest.util_trajectory_close().unwrap();
    }

    #[test]
    fn tng_io_testing() {
        // tng_io_testing.c:1296
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        let mut output_filename = input_filename.clone();
        input_filename.push(TEST_FILES_DIR);
        input_filename.push("tng_example.tng");
        output_filename.push(TEST_FILES_DIR);
        output_filename.push("tng_example_out.tng");

        let mut traj = Trajectory::new();
        traj.input_file_set(input_filename.as_path());
        traj.output_file_set(output_filename.as_path());

        test_read_and_write_file(&mut traj);

        // tng_io_testing.c:1306
        test_get_box_data(&mut traj);

        // tng_io_testing.c:1316 - Destroy and reinit trajectory
        drop(traj);
        let mut traj = Trajectory::new();

        let mut output_filename = std::env::current_dir().expect("able to get current working dir");
        output_filename.push(TEST_FILES_DIR);
        output_filename.push("tng_test.tng");
        traj.output_file_set(output_filename.as_path());

        // tng_io_testing.c:1329
        write_and_read_traj(&mut traj, USE_HASH);

        // tng_io_testing.c:1339
        get_positions_data(&mut traj, USE_HASH);

        // tng_io_testing.c:1360
        test_utility_functions(&mut traj, USE_HASH);

        // tng_io_testing.c:1371
        test_append(&mut traj, USE_HASH);

        // tng_io_testing.c:1381
        test_copy_container(&mut traj, USE_HASH);
    }
}

#[cfg(test)]
mod var_num_atoms_regression {
    use std::{
        fs,
        path::{Path, PathBuf},
        process,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::trajectory::Trajectory;

    fn unique_fixture_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tng_rs_var_num_atoms_frame_set_{}_{}.tng",
            process::id(),
            nanos
        ))
    }

    fn write_var_num_atoms_fixture(path: &Path) {
        let _ = fs::remove_file(path);

        let mut traj = Trajectory::new();
        traj.var_num_atoms = true;
        traj.output_file_set(path);

        let molecule_idx = traj.add_molecule("mono");
        let chain_idx = traj.add_chain(molecule_idx, "A");
        let residue_idx = traj.chain_residue_add(molecule_idx, chain_idx, "RES");
        traj.residue_atom_add(molecule_idx, residue_idx, "X", "X");

        traj.file_headers_write(false).unwrap();
        traj.frame_set_new(0, 1).unwrap();
        traj.current_trajectory_frame_set.molecule_cnt_list = vec![2];
        traj.frame_set_write(false).unwrap();
    }

    #[test]
    fn var_num_atoms_frame_set_counts_round_trip() {
        let path = unique_fixture_path();
        write_var_num_atoms_fixture(&path);

        let mut traj = Trajectory::new();
        traj.input_file_set(&path);
        traj.file_headers_read(false).unwrap();
        traj.frame_set_nr_find(0).unwrap();

        assert_eq!(traj.molecule_cnt_list_get().as_slice(), &[2]);
        assert_eq!(traj.num_particles_get(), 2);

        let _ = fs::remove_file(path);
    }
}

#[cfg(test)]
mod particle_lookup_regression {
    use crate::trajectory::Trajectory;

    #[test]
    fn residue_id_lookup_uses_residue_id_not_atom_id() {
        let mut traj = Trajectory::new();
        let molecule_idx = traj.add_molecule("mol");
        let chain_idx = traj.add_chain(molecule_idx, "A");
        let residue_idx = traj.chain_residue_add(molecule_idx, chain_idx, "RES");

        traj.residue_atom_add(molecule_idx, residue_idx, "A1", "A");
        traj.residue_atom_add(molecule_idx, residue_idx, "A2", "A");
        traj.molecule_cnt_set(molecule_idx, 1);

        assert_eq!(traj.residue_id_of_particle_nr_get(1), Some(0));
    }
}

#[cfg(test)]
mod bond_lookup_regression {
    use crate::trajectory::Trajectory;

    #[test]
    fn molsystem_bonds_offsets_repeated_molecules() {
        let mut traj = Trajectory::new();
        let molecule_idx = traj.add_molecule("mol");
        let chain_idx = traj.add_chain(molecule_idx, "A");
        let residue_idx = traj.chain_residue_add(molecule_idx, chain_idx, "RES");

        traj.residue_atom_add(molecule_idx, residue_idx, "A1", "A");
        traj.residue_atom_add(molecule_idx, residue_idx, "A2", "A");
        traj.add_molecule_bond(molecule_idx, 0, 1);
        traj.molecule_cnt_set(molecule_idx, 2);

        let (n_bonds, from_atoms, to_atoms) = traj.molsystem_bonds_get().unwrap();
        assert_eq!(n_bonds, 2);
        assert_eq!(from_atoms, vec![0, 2]);
        assert_eq!(to_atoms, vec![1, 3]);
    }
}

#[cfg(test)]
mod particle_number_lookup_regression {
    use crate::{MAX_STR_LEN, trajectory::Trajectory};

    #[test]
    fn particle_lookup_helpers_stop_at_first_matching_molecule() {
        let mut traj = Trajectory::new();

        let mol0 = traj.add_molecule("mol0");
        let chain0 = traj.add_chain(mol0, "A");
        let res0 = traj.chain_residue_add(mol0, chain0, "R0");
        traj.residue_atom_add(mol0, res0, "A0", "T0");
        traj.molecule_cnt_set(mol0, 1);

        let mol1 = traj.add_molecule("mol1");
        let chain1 = traj.add_chain(mol1, "B");
        let res1 = traj.chain_residue_add(mol1, chain1, "R1");
        traj.residue_atom_add(mol1, res1, "A1", "T1");
        traj.molecule_cnt_set(mol1, 1);

        traj.molecules[mol0].residues[0].id = 10;
        traj.molecules[mol1].residues[0].id = 20;

        assert_eq!(
            traj.molecule_name_of_particle_nr_get(0, MAX_STR_LEN)
                .unwrap(),
            "mol0"
        );
        assert_eq!(
            traj.chain_name_of_particle_nr_get(0, MAX_STR_LEN).unwrap(),
            "A"
        );
        assert_eq!(
            traj.residue_name_of_particle_nr_get(0, MAX_STR_LEN)
                .unwrap(),
            "R0"
        );
        assert_eq!(traj.residue_id_of_particle_nr_get(0), Some(10));
        assert_eq!(traj.global_residue_id_of_particle_nr_get(0), Some(10));
        assert_eq!(
            traj.atom_name_of_particle_nr_get(0, MAX_STR_LEN).unwrap(),
            "A0"
        );
        assert_eq!(traj.atom_type_of_particle_nr_get(0), "T0");
    }
}
