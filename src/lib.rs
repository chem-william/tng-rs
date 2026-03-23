#![allow(dead_code)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TngError {
    /// A constraint or validation was violated (e.g. stride length ordering).
    /// Corresponds to C's TNG_FAILURE for argument/state validation.
    #[error("{0}")]
    Constraint(String),

    /// An item was not found (e.g. molecule not in trajectory).
    #[error("{0}")]
    NotFound(String),

    /// A major, unspecified error has occured - matches the C code.
    #[error("{0}")]
    Critical(String),

    /// I/O error wrapping std::io::Error.
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
mod trajectory;
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
        MAX_STR_LEN,
        data::{Compression, DataType},
        gen_block::BlockID,
        molecule::Molecule,
        trajectory::Trajectory,
    };
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

    const USE_HASH: bool = false;

    /// C API: tng_test_setup_molecules() in tng_io_testing.c:45
    fn setup_molecules(traj: &mut Trajectory) {
        let mol_idx = traj.add_molecule("water");
        let chain_idx = traj.add_chain(mol_idx, "W");
        let residue_idx = traj.add_chain_residue(mol_idx, chain_idx, "WAT");
        let _o_idx = traj.add_residue_atom(mol_idx, residue_idx, "O", "O");
        let _h1_idx = traj.add_residue_atom(mol_idx, residue_idx, "HO1", "H");
        let _h2_idx = traj.add_residue_atom(mol_idx, residue_idx, "HO2", "H");
        let _bond_idx = traj.add_molecule_bond(mol_idx, 0, 1);
        let _bond_idx = traj.add_molecule_bond(mol_idx, 0, 2);

        traj.set_molecule_cnt(mol_idx, 200);
        let count = traj.get_molecule_cnt(mol_idx);
        assert_eq!(count, 200);
    }

    /// C API: tng_test_molecules() in tng_io_testing.c:141
    fn check_molecules(traj: &mut Trajectory) {
        let cnt = traj.num_molecules_types_get();
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
        traj.file_headers_read(USE_HASH);
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
    fn test_write_and_read_traj(traj: &mut Trajectory) -> Trajectory {
        traj.set_medium_stride_length(MEDIUM_STRIDE_LEN).unwrap();
        traj.set_long_stride_length(LONG_STRIDE_LEN).unwrap();

        traj.set_first_user_name(USER_NAME);
        traj.set_first_program_name(PROGRAM_NAME);
        traj.set_first_computer_name(COMPUTER_NAME);
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
        traj.add_data_block(
            BlockID::TrajBoxShape,
            "BOX SHAPE",
            DataType::Double,
            false,
            1,
            9,
            1,
            Compression::Uncompressed,
            Some(&bytes),
        )
        .unwrap();

        // Set the partial charges (treat the water as TIP3P)
        let n_particles = traj.get_num_particles();
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
            false,
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
            false,
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
        traj.add_data_block(
            BlockID::TrajGeneralComments,
            "COMMENTS",
            DataType::Char,
            false,
            1,
            1,
            1,
            Compression::Uncompressed,
            Some(annotation.as_bytes()),
        )
        .expect("Failed adding details annotation data block");

        // Write file headers (includes non trajectory data blocks)
        traj.file_headers_write(USE_HASH).unwrap();

        let n_frames_per_frame_set = traj.get_num_frames_per_frame_set();
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
                true,
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
        let mut traj = Trajectory::new();
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        input_filename.push(TEST_FILES_DIR);
        input_filename.push("tng_test.tng");
        traj.set_input_file(input_filename.as_path());

        traj.file_headers_read(USE_HASH);

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

        check_molecules(&mut traj);

        // TODO: particle_data_vector_get for masses (tng_io_testing.c:779-801)

        // Read all frame sets (tng_io_testing.c:804-842)
        while traj.frame_set_read_next(USE_HASH).is_ok() {}

        // TODO: remaining checks from tng_io_testing.c:844-922
        // - time_per_frame check
        // - frame_set_nr_find
        // - frame_set_read_current_only_data_from_block_id
        // - frame_set_read_next_only_data_from_block_id
        // - data_block_name_get, data_block_dependency_get
        // - data_block_num_values_per_frame_get, data_get_stride_length

        traj
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
    /// TODO: Port from C
    fn test_utility_functions(_traj: &mut Trajectory) {
        // TODO: port from tng_io_testing.c:1036-1140
        // - tng_util_trajectory_open (read mode)
        // - tng_util_time_of_frame_get for frames 50 and 100
        // - tng_util_num_frames_with_data_of_block_id_get
        // - tng_util_pos_read_range
        // - validate positions in range
        // - tng_util_trajectory_next_frame_present_data_blocks_find
        // - tng_util_frame_current_compression_get
        // - tng_util_trajectory_close
    }

    /// C API: tng_test_append() in tng_io_testing.c:1143
    /// TODO: Port from C
    fn test_append(_traj: &mut Trajectory) {
        // TODO: port from tng_io_testing.c:1143-1226
        // - tng_util_trajectory_open (append mode)
        // - set last_user_name, last_program_name, last_computer_name
        // - tng_file_headers_write
        // - tng_num_frames_get, tng_frame_set_of_frame_find
        // - tng_util_vel_with_time_double_write
        // - tng_util_trajectory_close
    }

    /// C API: tng_test_copy_container() in tng_io_testing.c:1228
    /// TODO: Port from C
    fn test_copy_container(_traj: &mut Trajectory) {
        // TODO: port from tng_io_testing.c:1228-1260
        // - tng_util_trajectory_open (read mode)
        // - tng_trajectory_init_from_src
        // - tng_molecule_system_copy
        // - close both trajectories
    }

    #[test]
    // #[ignore]
    fn tng_io_testing() {
        // tng_io_testing.c:1296
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        let mut output_filename = input_filename.clone();
        input_filename.push(TEST_FILES_DIR);
        input_filename.push("tng_example.tng");
        output_filename.push(TEST_FILES_DIR);
        output_filename.push("tng_example_out.tng");

        let mut traj = Trajectory::new();
        traj.set_input_file(input_filename.as_path());
        traj.set_output_file(output_filename.as_path());

        test_read_and_write_file(&mut traj);

        // tng_io_testing.c:1306
        test_get_box_data(&mut traj);

        // tng_io_testing.c:1316 - Destroy and reinit trajectory
        drop(traj);
        let mut traj = Trajectory::new();

        let mut output_filename = std::env::current_dir().expect("able to get current working dir");
        output_filename.push(TEST_FILES_DIR);
        output_filename.push("tng_test.tng");
        traj.set_output_file(output_filename.as_path());

        // tng_io_testing.c:1329
        let mut traj = test_write_and_read_traj(&mut traj);

        // tng_io_testing.c:1339
        get_positions_data(&mut traj, USE_HASH);

        // TODO: tng_io_testing.c:1360
        // test_utility_functions(&mut traj);

        // TODO: tng_io_testing.c:1371
        // test_append(&mut traj);

        // TODO: tng_io_testing.c:1381
        // test_copy_container(&mut traj);
    }
}
