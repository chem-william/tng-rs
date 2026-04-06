use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use tng_rs::{bench_data, bench_write};

fn bench_compress(c: &mut Criterion) {
    let output_path = std::env::temp_dir().join("tng_rs_criterion.tng");
    let (positions, velocities) = bench_data();

    c.bench_function("compress_100k_atoms_20_frames", |b| {
        b.iter(|| black_box(bench_write(&output_path, &positions, &velocities).unwrap()));
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(10));
    targets = bench_compress
}
criterion_main!(benches);
