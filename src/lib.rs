use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};
pub type Row = Matrix<f64, Const<1>, Dyn, VecStorage<f64, Const<1>, Dyn>>;
pub type Column = Matrix<f64, Dyn, Const<1>, VecStorage<f64, Dyn, Const<1>>>;

const MAX_REDUCED_COSTS: f64 = 0.0;
const TABLEAU_MIN_PIVOT_COL_PRIMAL_SIMPLEX: f64 = 0.00001;

fn get_reduced_costs(
    cb: &Row,
    cn: &Row,
    basis_inv: &DMatrix<f64>,
    non_basis_matrix: &DMatrix<f64>,
) -> Row {
    cb * basis_inv * non_basis_matrix - cn
}

fn get_entering_var(reduced_costs: &Row) -> Option<usize> {
    reduced_costs
        .iter()
        .enumerate()
        .filter(|(_, reduced_cost)| **reduced_cost < MAX_REDUCED_COSTS)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
}

fn get_leaving_var(
    a: &DMatrix<f64>,
    basis_inv: &DMatrix<f64>,
    b: &Column,
    entering_var: usize,
) -> Option<usize> {
    let tableau_entering_var_col = basis_inv * a.column(entering_var); //&non_basis_matrix.column(entering_var_loc);

    (basis_inv * b)
        .component_div(&tableau_entering_var_col)
        .iter()
        .enumerate()
        .filter(|(idx, _)| tableau_entering_var_col[*idx] >= TABLEAU_MIN_PIVOT_COL_PRIMAL_SIMPLEX)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) //Unwrap safe as 0 denominator filtered out
        .map(|(idx, _)| idx) // Extract the index of the minimum
}

pub fn primal_simplex(a: DMatrix<f64>, b: Column, c: Row, max_iterations: i32) -> Vec<usize> {
    let rows = a.nrows(); // Number of constraints
    let cols = a.ncols(); // Number of variables
    let mut basis_indices: Vec<usize> = (cols - rows..cols).collect();
    let mut non_basis_indices: Vec<usize> =
        (0..cols).filter(|i| !basis_indices.contains(&i)).collect();

    for it in 0..max_iterations {
        let non_basis_matrix = a.select_columns(&non_basis_indices);
        let basis_matrix = a.select_columns(&basis_indices);
        let cb = c.select_columns(&basis_indices);
        let cn = c.select_columns(&non_basis_indices);
        // TODO: add sherman morrison here
        let basis_inv_or_none = basis_matrix.try_inverse();
        let Some(basis_inv) = basis_inv_or_none else {
            print!("a is singular");
            break;
        };
        let reduced_costs = get_reduced_costs(&cb, &cn, &basis_inv, &non_basis_matrix);
        let entering_var_loc_or_none = get_entering_var(&reduced_costs);
        let Some(entering_var_loc) = entering_var_loc_or_none else {
            // Program terminated
            println!("{:?} iterations", it);
            break;
        };
        let entering_var = non_basis_indices[entering_var_loc];

        let leaving_var_loc_or_none =
            get_leaving_var(&a, &basis_inv, &b, entering_var);
        let Some(leaving_var_loc) = leaving_var_loc_or_none else {
            println!("Program is unbounded");
            break;
        };
        let leaving_var = basis_indices[leaving_var_loc];

        // need to replace leaving var with entering var (ie they are in same loc)
        // for sherman morrison algorithm to work
        basis_indices[leaving_var_loc] = entering_var;
        non_basis_indices[entering_var_loc] = leaving_var;
    }

    basis_indices
}

use serde::{Deserialize, Serialize};
use serde_json::Result; // Ensure these traits are imported

#[derive(Serialize, Deserialize)]
pub struct Matrices {
    pub a: DMatrix<f64>,
    pub b: Column,
    pub c: Row,
}

pub fn read_matrices_from_json(file_path: &str) -> Result<Matrices> {
    let data = std::fs::read_to_string(file_path).expect("data");
    let matrices: Matrices = serde_json::from_str(&data)?;
    Ok(matrices)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_small_example() {
        let small_matrices = read_matrices_from_json("./benches/data/smallExample.json")
            .expect("data should be here");
        let basis = primal_simplex(small_matrices.a, small_matrices.b, small_matrices.c, 10000);

        let expected_result = vec![1, 5, 6];
        assert_eq!(
            basis, expected_result,
            "Test failed: Expected [1, 5, 6], got {:?}",
            basis
        );
    }

    #[test]
    fn test_big_example() {
        let big_matrices = read_matrices_from_json("./benches/data/bigExample.json")
            .expect("data should be here");
        
        let mut basis = primal_simplex(big_matrices.a, big_matrices.b, big_matrices.c, 100);
        // TODO move
        let mut expected_result = vec![3, 6, 7, 15, 16, 17, 20, 25, 26, 30, 35, 37, 38, 39, 40, 42, 48, 50, 51, 55, 56, 61, 62, 64, 65, 67, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80, 82, 83, 84, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99];
        basis.sort();
        expected_result.sort();
        assert_eq!(
            basis, expected_result,
            "Test failed",
        );
    }
}
