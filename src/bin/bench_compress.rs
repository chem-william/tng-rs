use std::hint::black_box;
use tng_rs::{bench_data, bench_write};

fn main() {
    let output_path = std::env::temp_dir().join("tng_rs_flamegraph.tng");
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let (positions, velocities) = bench_data();

    for _ in 0..reps {
        black_box(bench_write(&output_path, &positions, &velocities).expect("write failed"));
    }
}
