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
) -> (usize, f64) {
    let (index, &value) = reduced_costs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .expect("Matrix should not be empty");

    (index, value)
}

fn get_leaving_var(
    non_basis_matrix: Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
    basis_inv: Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
    basis_indices: Vec<usize>,
    b: Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>,
    entering_loc: usize,
) -> (usize, usize) {
    let a_entering = &non_basis_matrix.column(entering_loc);
    let tableau = &basis_inv * a_entering;
    println!("{:?}", tableau);
    let basis_inv_b = basis_inv * b;
    let ratio = basis_inv_b.component_div(&tableau);
    println!("{:?}", ratio);
    let leaving_var_loc = ratio
        .iter()
        .enumerate()
        .filter(|(idx, _)| tableau[*idx] > 0.00001) // Dereference idx to get the value
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) // Unwrap safely if no NaN values
        .map(|(idx, _)| idx) // Extract the index of the minimum
        .expect("No valid indices found");
    let leaving_var = basis_indices[leaving_var_loc];

    (leaving_var_loc, leaving_var)
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
            let (entering_loc, entering_var) = get_entering_var(&reduced_costs);

            println!("{:?}", reduced_costs);
            println!("{:?}", entering_loc);
            println!("{:?}", entering_var);

            let (leaving_var_loc, leaving_var) =
                get_leaving_var(non_basis_matrix, basis_inv, basis_indices, b, entering_loc);
            println!("{:?}", leaving_var_loc);
            println!("{:?}", leaving_var);
        }
        None => {
            print!("not invertible")
        }
    }
}
