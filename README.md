[![Codecov](https://codecov.io/github/chem-william/tng-rs/coverage.svg?branch=main)](https://codecov.io/gh/chem-william/tng-rs)
[![dependency status](https://deps.rs/repo/github/chem-william/tng-rs/status.svg)](https://deps.rs/repo/github/chem-william/tng-rs)

# tng-rs

A Rust port of the Trajectory Next Generation (TNG) library. The original library was made by the GROMACS team and can be found [here](https://gitlab.com/gromacs/tng)

The port matches the behavior at commit `f8d55273` at [https://gitlab.com/gromacs/tng](https://gitlab.com/gromacs/tng).

 ## Notes
 Currently assumes that files have only been written on 64-bit systems.

 ## Development
 You can run the tests with debug output in the following way:
 ```bash
 RUST_LOG=debug cargo test -- --nocapture
 ```

 ## Testing
 There is internal unit testing to make sure that the Rust code works (and keeps working), there's FFI tests to compare with the C lib where the C code acts as the ground-truth (bugs and all),
 and there's some property-based testing using [`proptest`](https://docs.rs/proptest/latest/proptest/) where either the Rust code is expected to be internally consistent or the C code is used as a ground-truth.

 ## Differences from the C lib
 There are some patterns that are quite difficult to port where I decided to change the public API slightly. One being the fact that the original C lib makes use of circular references to have a molecule point to its parent residue while the parent residue points to that molecule. [It's not impossible to have in Rust](https://rust-classes.com/chapter_advanced_cicular_references), just a bit more cumbersome than in C.
Due to this, in some places the takes/returns an index to the residue, chain, molecules, etc. instead of the entity itself.

Searches over molecule-owned collections are exposed on `Molecule` in Rust. In practice, `chain_find` and `atom_find` correspond to C's `tng_molecule_chain_find` and `tng_molecule_atom_find`, while `residue_find` is a Rust convenience.

 ## LLM usage
 Parts of the code has been translated with the use of large language models (LLMs). All code has been reviewed by humans.
