#[macro_use]
extern crate ndarray;

use ndarray::{arr2, ArrayBase};
use ndarray::{Array1, Array2, Axis};

fn projection_space(space: ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 2]>>, projection: Array1<f64>) -> Array1<f64> {
    let space_shape = space.shape();
    let mut sum = Array1::<f64>::zeros(space_shape[0]);
    for (i, a) in space.axis_iter(Axis(0)).enumerate() {
        let proj = &a * &projection;
        let mut proj_sum = 0.;
        for c in 0..proj.shape()[0] {
            proj_sum += proj[c];
        }
        sum[i] = proj_sum;
    }
    sum
}


fn normalization(vector: &mut Array1<f64>) {
    let norm = vector.dot(vector).sqrt();
    for i in vector.iter_mut() {
        if norm.abs() >= 1e-8 {
            *i = *i / norm;
        }
    }
}

fn gram_schmidt(x: Array2<f64>) -> Array2<f64> {
    let shape = x.shape();
    let mut o = Array2::<f64>::zeros((shape[0], shape[1]));
    for i in 0..shape[1] {
        let vector = x.slice(s![..,i]);
        let space = o.slice(s![..,..i]);
        let projection = vector.dot(&space);
        let mut v2 = &vector - &projection_space(space, projection);
        normalization(&mut v2);
        for r in 0..shape[0] {
            o[(r, i)] = v2[r];
        }
    }
    o
}

fn print(vectors: &Array2<f64>) {
    for row in vectors.genrows() {
        for c in row {
            print!("{:>12.5}", c);
        }
        println!();
    }
    println!();
}

fn main() {
    let vectors = arr2(
        &[[1., 1., 2., 0., 1., 1.],
            [0., 0., 0., 1., 2., 1.],
            [1., 2., 3., 1., 3., 2.],
            [1., 0., 1., 0., 1., 1.]]);
    let orthonormal = gram_schmidt(vectors);
    println!("orthonormal:");
    print(&orthonormal);

    println!("orthonormal.t().dot(&orthonormal):");
    print(&orthonormal.t().dot(&orthonormal));

    let matrix = arr2(
        &[[1., 1., -1.],
            [1., 2., 1.],
            [1., 3., 0.]]);
    println!("QR decomposition:");
    let q = gram_schmidt(matrix.clone());
    println!("Q:");
    print(&q);
    let r = q.t().dot(&matrix);
    println!("R:");
    print(&r);
    println!("Q dot R:");
    print(&q.dot(&r));
}
