fn main() {
    let src = "vendor/tng/src/compression";
    let inc = "vendor/tng/include";

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
