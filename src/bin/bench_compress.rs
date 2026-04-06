use std::{hint::black_box, path::PathBuf, process::ExitCode};
use tng_rs::{bench_data, bench_write};

const DEFAULT_REPS: usize = 10;
const OUTPUT_FILE_NAME: &str = "tng_rs_flamegraph.tng";

fn main() -> ExitCode {
    let output_path = bench_output_path();
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPS);
    let (positions, velocities) = bench_data();
    let mut sink = 0u64;

    for _ in 0..reps {
        sink = black_box(bench_write(&output_path, &positions, &velocities).expect("write failed"));
    }

    ExitCode::from(u8::from(sink == 0))
}

fn bench_output_path() -> PathBuf {
    std::env::var_os("TMPDIR")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(OUTPUT_FILE_NAME)
}
