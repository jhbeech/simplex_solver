use simplex_solver::{primal_simplex, read_matrices_from_json};

fn main() {
    let mat =
        read_matrices_from_json("./benches/data/bigExample.json").expect("data should be here");
    let mut basis = primal_simplex(mat.a, mat.b, mat.c, 10000);
    basis.sort();
    assert_eq!(basis, mat.sol, "Test failed",);
}
