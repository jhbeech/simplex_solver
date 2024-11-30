use nalgebra::{DMatrix, DVector};

fn main() {
    let a = DMatrix::<f64>::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 6.0, 11.0]);
    let b = DVector::<f64>::from_vec(vec![1.0, 1.0]);
    let c = DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

    primal_simplex(a, b, c);
}

fn primal_simplex(a: DMatrix<f64>, _b: DVector<f64>, c: DVector<f64>) {
    let rows = a.nrows(); // Number of constraints
    let cols = a.ncols(); // Number of variables

    let basis_indices: Vec<usize> = (cols - rows..cols).collect();
    let non_basis_indices: Vec<usize> = (0..cols)
        .filter(|i| !basis_indices.contains(&i)) // Dereference `i` to pass a value to `contains`
        .collect();

    let basis_matrix = a.select_columns(&basis_indices);

    let basis_inv_or_none = basis_matrix.try_inverse();
    match basis_inv_or_none {
        Some(basis_inv) => {
            let non_basis_matrix = a.select_columns(&non_basis_indices);
            let cb = &c.transpose().select_columns(&basis_indices);
            let cn = &c.transpose().select_columns(&non_basis_indices);
            
            // Get entering var
            let reduced_costs = cb * &basis_inv * &non_basis_matrix - cn;
            let (entering_loc, entering_var) = reduced_costs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .expect("Matrix should not be empty");

            println!("{:?}", reduced_costs);
            println!("{:?}", entering_loc);
            println!("{:?}", entering_var);

            // Get leaving var

            // dont think this is technically the tableau
            let entering_matrix = &non_basis_matrix.column(entering_loc);
            let tableau = &basis_inv * entering_matrix;
            println!("{:?}", tableau);
            let basis_inv_b = basis_inv * _b;
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
            println!("{:?}", leaving_var_loc);
            println!("{:?}", leaving_var);
        }
        None => {
            print!("not invertible")
        }
    }
}
