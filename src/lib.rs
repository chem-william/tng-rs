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

// The maximum allowed length of a string
const MAX_STR_LEN: usize = 1024;

// The length of an MD5 has
const MD5_HASH_LEN: usize = 16;

const API_VERSION: u64 = 8;

#[cfg(test)]
mod integration {
    use crate::trajectory::Trajectory;

    use super::*;

    use std::env;
    use std::path::Path;
    use std::process::exit;

    #[test]
    fn it_works() {
        let filename = Path::new("/home/william/workspace/rust/tng-rs/tng_test.tng");
        let mut traj = Trajectory::new();

        // Tell the library which file to open
        traj.set_input_file(filename);

        // Read file headers
        traj.file_headers_read();

        // if tng_file_headers_read(&mut traj, TNG_USE_HASH) != TNG_SUCCESS {
        //     eprintln!("tng_file_headers_read failed");
        //     tng_trajectory_destroy(traj);
        //     exit(1);
        // }

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
