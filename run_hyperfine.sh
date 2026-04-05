#!/usr/bin/env bash
set -euo pipefail

SRC=vendor/tng/src/compression
INC=vendor/tng/include
LIB=vendor/tng/src/lib
BININC=vendor/tng/build/include
ZLIB=vendor/tng/external/zlib
REPS=10

gcc -std=c11 -O2 \
  c_test/bench_compress.c \
  "$LIB"/md5.c "$LIB"/tng_io.c \
  "$SRC"/bwlzh.c "$SRC"/bwt.c "$SRC"/coder.c "$SRC"/dict.c \
  "$SRC"/fixpoint.c "$SRC"/huffman.c "$SRC"/huffmem.c "$SRC"/lz77.c \
  "$SRC"/merge_sort.c "$SRC"/mtf.c "$SRC"/rle.c "$SRC"/tng_compress.c \
  "$SRC"/vals16.c "$SRC"/warnmalloc.c "$SRC"/widemuldiv.c \
  "$SRC"/xtc2.c "$SRC"/xtc3.c \
  "$ZLIB"/adler32.c "$ZLIB"/compress.c "$ZLIB"/crc32.c "$ZLIB"/deflate.c \
  "$ZLIB"/inffast.c "$ZLIB"/inflate.c "$ZLIB"/inftrees.c "$ZLIB"/trees.c \
  "$ZLIB"/uncompr.c "$ZLIB"/zutil.c \
  -I"$INC" -I"$BININC" -I"$ZLIB" -DUSE_STD_INTTYPES_H -lm -o c_test/bench_compress

cargo build --release --bin bench_compress

hyperfine --warmup 3 \
  -n C    "./c_test/bench_compress $REPS" \
  -n Rust "./target/release/bench_compress $REPS"
