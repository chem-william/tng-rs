# if a glibc error pops up, just run `cargo clean` - see https://github.com/cross-rs/cross/issues/724
# powerpc64 is big-endian 64-bit
cross test --target powerpc64-unknown-linux-gnu
# powerpc is big-endian 32-bit
# cross test --target powerpc-unknown-linux-gnu
# armv7 is little-endian 32-bit
# cross test --target armv7-unknown-linux-gnueabihf
