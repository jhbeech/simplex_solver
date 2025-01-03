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

fn get_inv_sherman_morrison(
    a: &DMatrix<f64>,
    basis_inv: &DMatrix<f64>,
    entering_var: usize,
    leaving_var: usize,
    leaving_var_loc: usize,
) -> DMatrix<f64> {
    let u = Column::from_vec(
        (0..a.nrows())
            .map(|i| a[(i, entering_var)] - a[(i, leaving_var)])
            .collect::<Vec<f64>>(),
    );
    let v = Column::from_vec(
        (0..a.nrows())
            .map(|i| if i == leaving_var_loc { 1.0 } else { 0.0 })
            .collect::<Vec<f64>>(),
    );
    let numerator = (basis_inv * &u) * (&v.transpose() * basis_inv);
    let denominator = (v.transpose() * basis_inv * &u)[(0,0)] + 1.0;
    let update = - numerator / denominator;

    basis_inv + update
}

pub fn primal_simplex(a: DMatrix<f64>, b: Column, c: Row, max_iterations: i32) -> Vec<usize> {
    let rows = a.nrows(); // Number of constraints
    let cols = a.ncols(); // Number of variables
    let mut basis_indices: Vec<usize> = (cols - rows..cols).collect();
    let mut non_basis_indices: Vec<usize> =
        (0..cols).filter(|i| !basis_indices.contains(&i)).collect();
    let basis_inv_or_none = a.select_columns(&basis_indices).try_inverse();
    let Some(mut basis_inv) = basis_inv_or_none else {
        print!("a is singular");
        return vec![];
    };
    
    for it in 0..max_iterations {
        let non_basis_matrix = a.select_columns(&non_basis_indices);
        let cb = c.select_columns(&basis_indices);
        let cn = c.select_columns(&non_basis_indices);
        let reduced_costs = get_reduced_costs(&cb, &cn, &basis_inv, &non_basis_matrix);
        let entering_var_loc_or_none = get_entering_var(&reduced_costs);
        let Some(entering_var_loc) = entering_var_loc_or_none else {
            // Program terminated
            println!("{:?} iterations", it);
            break;
        };
        let entering_var = non_basis_indices[entering_var_loc];
        
        let leaving_var_loc_or_none = get_leaving_var(&a, &basis_inv, &b, entering_var);
        let Some(leaving_var_loc) = leaving_var_loc_or_none else {
            println!("Program is unbounded");
            break;
        };
        let leaving_var = basis_indices[leaving_var_loc];
        
        print!("it: {}, leaving: {},entering {}\n", it, leaving_var, entering_var);

        basis_indices[leaving_var_loc] = entering_var;
        non_basis_indices[entering_var_loc] = leaving_var;
        basis_inv = get_inv_sherman_morrison(&a, &basis_inv, entering_var, leaving_var, leaving_var_loc);
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
    pub sol: Vec<usize>
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
        let mut basis = primal_simplex(small_matrices.a, small_matrices.b, small_matrices.c, 10000);
        basis.sort();
        let expected_result = small_matrices.sol;
        assert_eq!(
            basis, expected_result,
            "Small test failed",
        );
    }

    #[test]
    fn test_big_example() {
        let big_matrices =
            read_matrices_from_json("./benches/data/bigExample.json").expect("data should be here");

        let mut basis = primal_simplex(big_matrices.a, big_matrices.b, big_matrices.c, 100);
        let expected_result = big_matrices.sol;
        basis.sort();
        assert_eq!(basis, expected_result, "Test failed",);
    }
}
