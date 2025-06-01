mod atom;
mod bond;
mod chain;
mod data;
mod gen_block;
mod molecule;
mod particle_mapping;
mod residue;
mod trajectory;
mod trajectory_frame_set;
mod utils;

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
        FRAME_DEPENDENT, PARTICLE_DEPENDENT, gen_block::BlockID, molecule::Molecule,
        trajectory::Trajectory,
    };
    use assert_approx_eq::assert_approx_eq;

    const N_FRAME_SETS: i64 = 100;
    const TIME_PER_FRAME: f64 = 2e-15;

    #[test]
    fn can_we_init_traj_with_time() {
        let traj = Trajectory::new();
        assert!(traj.time > 100);
    }

    #[test]
    fn test_read_write() {
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        let mut output_filename = input_filename.clone();
        input_filename.push("tng_example.tng");
        output_filename.push("tng_example_out.tng");

        let mut traj = Trajectory::new();

        // Tell the library which files to open
        traj.set_input_file(input_filename.as_path());
        traj.set_output_file(output_filename.as_path());

        // test_read_and_write_file

        assert_eq!(traj.input_file_path, input_filename);
        assert_eq!(traj.output_file_path, output_filename);

        traj.file_headers_read();
        // traj.file_headers_write();
    }

    #[test]
    fn it_works() {
        let mut input_filename = std::env::current_dir().expect("able to get current working dir");
        input_filename.push("tng_test.tng");
        let mut traj = Trajectory::new();

        // Tell the library which file to open
        traj.set_input_file(input_filename.as_path());

        // Read file headers
        traj.file_headers_read();

        assert_eq!(traj.first_user_name, "USER 1");
        assert_eq!(traj.first_program_name, "tng_testing");
        assert_eq!(traj.first_computer_name, "Unknown computer");
        assert_eq!(traj.forcefield_name, "No forcefield");
        assert_eq!(traj.medium_stride_length, 5);
        assert_eq!(traj.long_stride_length, 25);
        assert_eq!(traj.compression_precision, 1000.0);
        assert_eq!(traj.distance_unit_exponential, -9);

        // Test molecule properties
        assert_eq!(traj.n_molecules, 1);
        assert_eq!(traj.molecule_cnt_list[0], 200);
        assert!(!traj.var_num_atoms);
        let molecule = &traj.molecules[0];

        assert!(traj.find_molecule("water", -1).is_some());

        assert_eq!(molecule.name, "water");

        // num_chains_get
        assert_eq!(molecule.n_chains, 1);

        // chain_of_index_get
        let _chain = &molecule.chains[0];

        // chain_find
        let chain = molecule
            .chain_find("W", -1)
            .expect("'W' chain to be present");

        // num_residues_get
        assert_eq!(molecule.n_residues, 1);

        // residue_of_index_get
        let _residue = &molecule.residues[0];

        // num_atoms_get
        assert_eq!(molecule.n_atoms, 3);

        // atom_of_index_get
        let _atom = &molecule.atoms[0];
        molecule.atom_find("O", -1).expect("'O' to be present");

        // chain_name_get
        assert_eq!(&chain.name, "W");

        // chain_num_residues_get
        assert_eq!(chain.n_residues, 1);

        // chain_residue_of_index_get
        let _chain_residue = &molecule.residues[chain.residues_indices.0];

        // chain_residue_find
        let residue = &molecule
            .residue_find("WAT", -1)
            .expect("residue on molecule");

        // residue_name_get
        assert_eq!(residue.name, "WAT");

        // residue_num_atoms_get
        assert_eq!(residue.n_atoms, 3);

        // residue_atom_of_index_get
        let atom_of_residue = molecule.residue_atom_of_index(0, residue);

        // atom_name_get
        assert_eq!(atom_of_residue.name, "O");

        // atom_type_get
        assert_eq!(atom_of_residue.atom_type, "O");

        // molecule_id_of_particle_nr_get
        assert_eq!(traj.molecule_id_of_particle_nr_get(0), Some(1));

        // residue_id_of_particle_nr_get
        assert_eq!(traj.residue_id_of_particle_nr_get(0), Some(0));

        // global_residue_id_of_particle_nr_get
        assert_eq!(traj.global_residue_id_of_particle_nr_get(599), Some(199));

        // molecule_name_of_particle_nr_get
        assert_eq!(traj.molecule_name_of_particle_nr_get(0), "water");

        // chain_name_of_particle_nr_get
        assert_eq!(traj.chain_name_of_particle_nr_get(0), "W");

        // residue_name_of_particle_nr_get
        assert_eq!(traj.residue_name_of_particle_nr_get(0), "WAT");

        // atom_name_of_particle_nr_get
        assert_eq!(traj.atom_name_of_particle_nr_get(0), "O");

        // molecule_alloc
        let mut molecule = Molecule::new();
        molecule.name = "TEST".to_string();
        traj.molecule_existing_add(molecule);

        // molsystem_bonds_get
        let (bonds, from_atoms, to_atoms) =
            traj.molsystem_bonds_get().expect("molsystem to have bonds");
        assert_eq!(bonds, 400);
        assert_eq!(from_atoms.len(), 400);
        assert_eq!(to_atoms.len(), 400);

        // particle_data_vector_get
        let (_read_n_particles, masses) = traj
            .particle_data_vector(true, BlockID::TrajMasses)
            .expect("particle data");
        // TODO
        // assert_eq!(read_n_particles, n_particles);

        // Above we have written only water molecules (in the order oxygen, hydrogen, hydrogen ...).
        // Test that the first and second as well as the very last atoms (oxygen, hydrogen and hydrogen)
        assert_approx_eq!(masses[0], 16.0);
        assert_approx_eq!(masses[1], 1.008);
        assert_approx_eq!(masses.last().unwrap(), 1.008);

        let mut i = 0;
        loop {
            let result = traj.frame_set_read_next();
            if result.is_err() {
                break;
            }
            let frame_set = &traj.current_trajectory_frame_set;
            // temp_int
            let prev_frame_set_file_pos = frame_set.prev_frame_set_file_pos;
            // temp_int2
            let next_frame_set_file_pos = frame_set.next_frame_set_file_pos;

            if i > 0 {
                if prev_frame_set_file_pos == -1 {
                    panic!("file position of previous frame set not correct");
                }
            } else if prev_frame_set_file_pos != -1 {
                panic!("file position of previous frame set not correct");
            }

            if i < N_FRAME_SETS - 1 {
                if next_frame_set_file_pos == -1 {
                    panic!("file position of next frame set not correct");
                }
            } else if next_frame_set_file_pos != -1 {
                panic!("file position of previouss next set not correct");
            }
            i += 1;
        }

        assert_approx_eq!(traj.time_per_frame, TIME_PER_FRAME);

        assert!(traj.frame_set_nr_find(N_FRAME_SETS * 3 / 10).is_ok());
        assert!(traj.frame_set_nr_find(N_FRAME_SETS * 3 / 4).is_ok());

        // frame_set_get
        // frame_set_frame_range_get
        let current_frame_set = &traj.current_trajectory_frame_set;
        let first_frame = current_frame_set.first_frame;
        assert_eq!(first_frame, 75 * traj.frame_set_n_frames);

        assert!(
            traj.frame_set_read_current_only_data_from_block_id(BlockID::TrajPositions)
                .is_ok()
        );
        assert!(
            traj.frame_set_read_next_only_data_from_block_id(BlockID::TrajPositions)
                .is_ok()
        );

        let data_block_name = traj.data_block_name_get(BlockID::TrajPositions);
        assert_eq!(data_block_name, Ok("POSITIONS".to_string()));

        let data_block_name = traj.data_block_name_get(BlockID::TrajForces);
        assert_eq!(data_block_name, Err(()));

        let dependency = traj.data_block_dependency_get(BlockID::TrajPositions);
        assert_eq!(dependency, Ok(FRAME_DEPENDENT + PARTICLE_DEPENDENT));

        let n_values_per_frame = traj.data_block_num_values_per_frame_get(BlockID::TrajPositions);
        assert_eq!(n_values_per_frame, Ok(3));

        let result = traj.data_get_stride_length(BlockID::TrajPositions, 100);
        assert_eq!(result, Ok(1));

        // // How many frames in the file?
        // let mut tot_n_frames: i64 = 0;
        // if tng_num_frames_get(&traj, &mut tot_n_frames) != TNG_SUCCESS {
        //     eprintln!("Cannot determine the number of frames in the file");
        //     tng_trajectory_destroy(traj);
        //     exit(1);
        // }
        // println!("{} frames in file", tot_n_frames);

        // // Clamp last_frame
        // if (last_frame as i64) > tot_n_frames - 1 {
        //     last_frame = (tot_n_frames - 1) as i32;
        // }
        // let n_frames: i32 = last_frame - first_frame + 1;

        // // Buffers for names (64 bytes each, zeroed)
        // let mut atom_buf: [u8; 64] = [0; 64];
        // let mut res_buf: [u8; 64] = [0; 64];
        // let mut chain_buf: [u8; 64] = [0; 64];

        // let got_atom = tng_atom_name_of_particle_nr_get(&traj, particle, &mut atom_buf);
        // let got_res = tng_residue_name_of_particle_nr_get(&traj, particle, &mut res_buf);
        // let got_chain = tng_chain_name_of_particle_nr_get(&traj, particle, &mut chain_buf);

        // if got_atom == TNG_SUCCESS && got_res == TNG_SUCCESS && got_chain == TNG_SUCCESS {
        //     // Convert first null‐terminated segment of each buffer into a Rust &str
        //     let atom_name = String::from_utf8_lossy(
        //         &atom_buf[..atom_buf.iter().position(|&b| b == 0).unwrap_or(64)],
        //     );
        //     let res_name = String::from_utf8_lossy(
        //         &res_buf[..res_buf.iter().position(|&b| b == 0).unwrap_or(64)],
        //     );
        //     let chain_name = String::from_utf8_lossy(
        //         &chain_buf[..chain_buf.iter().position(|&b| b == 0).unwrap_or(64)],
        //     );
        //     println!("Particle: {} ({}: {})", atom_name, chain_name, res_name);
        // } else {
        //     println!("Particle name not found");
        // }

        // // Prepare to receive a 3D Vec: frames × particles × values
        // let mut positions: Vec<Vec<Vec<DataValues>>> = Vec::new();
        // let mut n_particles: i64 = 0;
        // let mut n_values_per_frame: i64 = 0;
        // let mut data_type: char = '\0';

        // let rc = tng_particle_data_interval_get(
        //     &traj,
        //     TNG_TRAJ_POSITIONS,
        //     first_frame,
        //     last_frame,
        //     TNG_USE_HASH,
        //     &mut positions,
        //     &mut n_particles,
        //     &mut n_values_per_frame,
        //     &mut data_type,
        // );

        // if rc == TNG_SUCCESS {
        //     if particle >= n_particles {
        //         println!(
        //             "Chosen particle out of range. Only {} particles in trajectory.",
        //             n_particles
        //         );
        //     } else {
        //         // For each frame, print the frame index + each value
        //         for (frame_idx, frame_data) in positions.iter().enumerate() {
        //             print!("{}", first_frame + frame_idx as i32);
        //             // frame_data: Vec<Vec<DataValues>>, indexed by [particle][value_idx]
        //             let particle_row = &frame_data[particle as usize];
        //             for val in particle_row.iter() {
        //                 match (data_type, val) {
        //                     (TNG_INT_DATA, DataValues::Int(i)) => {
        //                         print!("\t{}", i);
        //                     }
        //                     (TNG_FLOAT_DATA, DataValues::Float(f)) => {
        //                         print!("\t{}", f);
        //                     }
        //                     (TNG_DOUBLE_DATA, DataValues::Double(d)) => {
        //                         print!("\t{}", d);
        //                     }
        //                     _ => {}
        //                 }
        //                 println!();
        //             }
        //         }
        //     }
        // } else {
        //     println!("Cannot read positions");
        // }

        // // Free the positions memory
        // if !positions.is_empty() {
        //     tng_particle_data_values_free(
        //         traj,
        //         positions,
        //         n_frames,
        //         n_particles,
        //         n_values_per_frame,
        //         data_type,
        //     );
        // }

        // tng_trajectory_destroy(traj);
    }
}
