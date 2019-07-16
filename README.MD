Day 66: Gram-Schmidt
<br>
Gram-Schmidt orthogonalization is a method used in linear algebra to produce an orthogonal [or orthonormal] base that generates the same vector space as a linear span of a given set of vectors.
<br>
Gram-Schmidt also gives us QR decomposition for free. It is a process of decomposing matrix X into a product of two matrices, X = QR, where Q is an orthogonal matrix and R is upper triangular matrix.
<br>
run:

```

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

```

result:

```

orthonormal:
     0.57735    -0.00000    -0.00000    -0.30861     0.00000    -0.00000
     0.00000     0.00000     0.00000     0.92582     0.00000     0.00000
     0.57735     0.70711     0.00000     0.15430     0.00000     0.00000
     0.57735    -0.70711    -0.00000     0.15430    -0.00000    -0.00000

orthonormal.t().dot(&orthonormal):
     1.00000    -0.00000    -0.00000    -0.00000     0.00000    -0.00000
    -0.00000     1.00000     0.00000     0.00000     0.00000     0.00000
    -0.00000     0.00000     0.00000     0.00000     0.00000     0.00000
    -0.00000     0.00000     0.00000     1.00000     0.00000     0.00000
     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000
    -0.00000     0.00000     0.00000     0.00000     0.00000     0.00000

QR decomposition:
Q:
     0.57735    -0.70711    -0.40825
     0.57735    -0.00000     0.81650
     0.57735     0.70711    -0.40825

R:
     1.73205     3.46410     0.00000
    -0.00000     1.41421     0.70711
     0.00000     0.00000     1.22474

Q dot R:
     1.00000     1.00000    -1.00000
     1.00000     2.00000     1.00000
     1.00000     3.00000    -0.00000

```