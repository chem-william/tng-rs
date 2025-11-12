# if a glibc error pops up, just run `cargo clean` - see https://github.com/cross-rs/cross/issues/724
# powerpc should be big-endian 
cross test --target powerpc64-unknown-linux-gnu
