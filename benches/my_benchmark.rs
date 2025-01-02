use criterion::{criterion_group, criterion_main, Criterion};
use simplex_solver::{primal_simplex, read_matrices_from_json};

fn small_example() {
    let small_matrices = read_matrices_from_json("./benches/data/smallExample.json").expect("data should be here");
    primal_simplex(
        small_matrices.a,
        small_matrices.b,
        small_matrices.c,
        10000
    );
}

fn big_example() {
    let small_matrices = read_matrices_from_json("./benches/data/bigExample.json").expect("data should be here");
    primal_simplex(
        small_matrices.a,
        small_matrices.b,
        small_matrices.c,
        1000000000
    );
}

fn benchmark_small(c: &mut Criterion) {
    c.bench_function("Big example", |b| b.iter(|| small_example()));
}

fn benchmark_big(c: &mut Criterion) {
    c.bench_function("Big example", |b| b.iter(|| big_example()));
}

criterion_group!(benches, benchmark_small, benchmark_big);
criterion_main!(benches);