use simplex_solver::{primal_simplex, read_matrices_from_json};

fn main() {
    let mat =
        read_matrices_from_json("./benches/data/bigExample.json").expect("data should be here");
    let mut basis = primal_simplex(mat.a, mat.b, mat.c, 100);
    println!("{:?}", basis);
    // TODO move
    let mut expected_result = vec![
        3, 6, 7, 15, 16, 17, 20, 25, 26, 30, 35, 37, 38, 39, 40, 42, 48, 50, 51, 55, 56, 61, 62,
        64, 65, 67, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80, 82, 83, 84, 87, 88, 90, 91, 92, 94,
        95, 96, 97, 99,
    ];
    println!("{:?}", basis);
    basis.sort();
    expected_result.sort();
    assert_eq!(basis, expected_result, "Test failed",);
}
