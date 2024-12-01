use nalgebra::{DMatrix, DVector};

fn main() {
    let a = DMatrix::<f64>::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 6.0, 11.0]);
    let b = DVector::<f64>::from_vec(vec![1.0, 1.0]);
    let c = DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

    primal_simplex(a, b, c);
}

use nalgebra::{Const, Dynamic, Matrix, VecStorage};

fn get_reduced_costs(
    cb: &Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    cn: &Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
    basis_inv: &Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
    non_basis_matrix: &Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
) -> Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>> {
    cb * basis_inv * non_basis_matrix - cn
}

fn get_entering_var(
    reduced_costs: &Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>,
) -> Option<usize> {
    reduced_costs
        .iter()
        .enumerate()
        .filter(|(_, reduced_cost)| **reduced_cost < 0.0)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
}

fn get_leaving_var(
    non_basis_matrix: Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
    basis_inv: Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
    b: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    entering_loc: usize,
) -> Option<usize> {
    let a_entering = &non_basis_matrix.column(entering_loc);
    let tableau = &basis_inv * a_entering;
    let basis_inv_b = basis_inv * b;
    let ratio = basis_inv_b.component_div(&tableau);
    let leaving_var_loc = ratio
        .iter()
        .enumerate()
        .filter(|(idx, _)| tableau[*idx] > 0.00001) // Dereference idx to get the value
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) // Unwrap safely if no NaN values
        .map(|(idx, _)| idx); // Extract the index of the minimum
    leaving_var_loc
}

fn primal_simplex(a: DMatrix<f64>, b: DVector<f64>, c: DVector<f64>) {
    let rows = a.nrows(); // Number of constraints
    let cols = a.ncols(); // Number of variables
    let basis_indices: Vec<usize> = (cols - rows..cols).collect();
    let non_basis_indices: Vec<usize> = (0..cols).filter(|i| !basis_indices.contains(&i)).collect();
    let basis_matrix = a.select_columns(&basis_indices);
    let basis_inv_or_none = basis_matrix.try_inverse();
    match basis_inv_or_none {
        Some(basis_inv) => {
            let non_basis_matrix = a.select_columns(&non_basis_indices);
            let cb = c.transpose().select_columns(&basis_indices);
            let cn = c.transpose().select_columns(&non_basis_indices);
            let reduced_costs = get_reduced_costs(&cb, &cn, &basis_inv, &non_basis_matrix);
            let entering_var_loc_or_none = get_entering_var(&reduced_costs);
            match entering_var_loc_or_none {
                Some(entering_var_loc) => {
                    let entering_var = non_basis_indices[entering_var_loc];
                    let leaving_var_loc_or_none =
                        get_leaving_var(non_basis_matrix, basis_inv, b, entering_var_loc);
                    match leaving_var_loc_or_none {
                        Some(leaving_var_loc) => {
                            let leaving_var = basis_indices[leaving_var_loc];
                        }
                        None => {
                            // Unbounded program
                        }
                    }
                }
                None => {
                    // Program terminated
                }
            }
        }
        None => {
            // Singular matrix
        }
    }
}
