fn main() {
    let src = "vendor/tng/src/compression";
    let inc = "vendor/tng/include";

    // Skip C compilation when vendor sources are not present (e.g. on crates.io).
    // The C library is only used by #[cfg(test)] comparison tests.
    if !std::path::Path::new(src).exists() {
        return;
    }

    cc::Build::new()
        .std("c11")
        .include(inc)
        .define("USE_STD_INTTYPES_H", None)
        .warnings(false)
        .files(
            [
                "bwlzh.c",
                "bwt.c",
                "coder.c",
                "dict.c",
                "fixpoint.c",
                "huffman.c",
                "huffmem.c",
                "lz77.c",
                "merge_sort.c",
                "mtf.c",
                "rle.c",
                "tng_compress.c",
                "vals16.c",
                "warnmalloc.c",
                "widemuldiv.c",
                "xtc2.c",
                "xtc3.c",
            ]
            .iter()
            .map(|f| format!("{src}/{f}")),
        )
        .compile("tng_compression");

    #[cfg(unix)]
    println!("cargo:rustc-link-lib=m");
}
