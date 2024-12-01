use nalgebra::{Const, Dynamic, DMatrix, DVector, Matrix, VecStorage};
type Row = Matrix<f64, Const<1>, Dynamic, VecStorage<f64, Const<1>, Dynamic>>;
type Column = Matrix<f64, Dynamic, Const<1>, VecStorage<f64, Dynamic, Const<1>>>;

fn main() {
    let a = DMatrix::<f64>::from_row_slice(
        3,
        7,
        &[
            5., 2., 5., 10., 1., 0., 0., 10., 0., 3., 1., 0., 1., 0., 1., 1., 1., 2., 0., 0., 1.,
        ],
    );
    let b = DVector::<f64>::from_vec(vec![10., 12., 25.]);
    let c = DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0, 0., 0., 0.]).transpose();
    primal_simplex(a, b, c, 100);
}


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
        .filter(|(_, reduced_cost)| **reduced_cost < 0.0)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
}

fn get_leaving_var(
    non_basis_matrix: &DMatrix<f64>,
    basis_inv: &DMatrix<f64>,
    b: &Column,
    entering_loc: usize,
) -> Option<usize> {
    let tableau_row = basis_inv * &non_basis_matrix.column(entering_loc);
    (basis_inv * b)
        .component_div(&tableau_row)
        .iter()
        .enumerate()
        .filter(|(idx, _)| tableau_row[*idx] > 0.00001)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) //Unwrap safe as 0 denominator filtered out
        .map(|(idx, _)| idx) // Extract the index of the minimum
}

fn primal_simplex(a: DMatrix<f64>, b: Column, c: Row, max_iterations: i32) {
    let rows = a.nrows(); // Number of constraints
    let cols = a.ncols(); // Number of variables
    let mut basis_indices: Vec<usize> = (cols - rows..cols).collect();

    for _ in 0..max_iterations {
        let non_basis_indices: Vec<usize> =
            (0..cols).filter(|i| !basis_indices.contains(&i)).collect();
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
            // Terminated program
            break;
        };
        let entering_var = non_basis_indices[entering_var_loc];

        let leaving_var_loc_or_none =
            get_leaving_var(&non_basis_matrix, &basis_inv, &b, entering_var_loc);
        let Some(leaving_var_loc) = leaving_var_loc_or_none else {
            println!("Program is unbounded");
            break;
        };
        let _leaving_var = basis_indices[leaving_var_loc];

        // need to replace leaving var with entering var (ie they are in same loc)
        // for sherman morrison algorithm to work
        basis_indices[leaving_var_loc] = entering_var;
        // non_basis_indices[entering_var_loc] = leaving_var;
    }

    println!("{:?}", basis_indices);
}
