## Linear Algebra
*	Matrix multiplication is not commutative unlike scalar multiplication:

	$$$ \boldsymbol{AB} \neq \boldsymbol{BA} $$$

*	The dot product between 2 vectors is cumulative:

	$$$ \boldsymbol{x}^{T}\boldsymbol{y} = \boldsymbol{y}^{T}\boldsymbol{x} $$$
	
*	The transpose of a matrix product has a simple form:

	$$$ (\boldsymbol{AB})^{T} = \boldsymbol{B}^{T} \boldsymbol{A}^{T} $$$	
---

** NB 1: **
Computing $$$ \boldsymbol{A^{-1}} $$$ for different values of $$$\boldsymbol{b}$$$ in a $$$ \boldsymbol{x} = \boldsymbol{A^{-1}b}  $$$ solution to a $$$ \boldsymbol{Ax = b} $$$ equation is not recommended in practice because of limited precision on a digital computer. Algorithms that make use of the value of $$$ \boldsymbol{b} $$$ lead to higher accuracy.

** NB 2: **
Linear equations can only have none, one or infinitely many solutions. If both $$$\boldsymbol{x}$$$ and $$$\boldsymbol{y}$$$  are solutions, then: $$$ \boldsymbol{z} = \alpha\boldsymbol{x} + (1 - \alpha)\boldsymbol{y} $$$ is also a solution for any real $$$\alpha$$$.

** NB 3: **
To analyze how many solutions the equation $$$ \boldsymbol{Ax = b} $$$ has, think of the columns of $$$ \boldsymbol{A} $$$ as specifying different directions we can travel in from the origin, then determine how many ways there are of reaching $$$ \boldsymbol{b} $$$. This leads to the following equation:

$$ \boldsymbol{Ax} = \sum\_{i}{x_i\boldsymbol{A}_{:,i}} $$

** NB 4: **
For $$$\boldsymbol{A}$$$ to have an inverse, equation: $$$ \boldsymbol{Ax = b} $$$ should have one solution for every value of $$$ \boldsymbol{b} $$$. This means $$$ \boldsymbol{A} $$$ should be a square matrix with linearly independent columns.

** NB 5: **
A square matrix with linearly dependent columns is known as **singular** or any of the eigenvalues are zeros.

** NB 6: **
The squared $$$L^{2}$$$ norm is more convenient to work with mathematically: for example, each derivative of the squared $$$L^{2}$$$ norm with respect to each element of $$$ \boldsymbol{x} $$$ depends only on the corresponding element of $$$ \boldsymbol{x} $$$, while all the derivatives of the $$$L^{2}$$$ norm depend on the entire vector.

However, the squared $$$L^{2}$$$ norm may be undesirable because it increases very slowly near the origin. In several machine learning applications, it is important to discriminate between elements that are exactly zero and elements that are small but nonzero.

The $$$ L^{1} $$$ norm is commonly used in machine learning when the difference between zero and nonzero elements is very important.

Sometimes we wish to measure the size of a matrix via **Frobenius norm:**

$$ {\|\|A\|\|}\_{2} = \sqrt{  \sum\_{i,j}{ A^2_{i,j} }  } $$ 

---

*	** Orthonormal matrix ** is square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal:

	$$ \boldsymbol{A}^T\boldsymbol{A} = \boldsymbol{A}\boldsymbol{A}^T = \boldsymbol{I} $$
	
	That means:
	
	$$ \boldsymbol{A}^{-1} = \boldsymbol{A}^T $$
	
	
*	** Eigendecomposition **

Not every matrix can be decomposed into eigenvalues and eigenvectors. In some cases, the decomposition exists but involves complex rather than real numbers.

*	Case of a matrix $$$ \boldsymbol{A} $$$ that has $$$ \mathit{n} $$$ linearly independent eigenvectors. We can write the 	matrix like so:

	$$ \boldsymbol{A} = \boldsymbol{V}diag(\boldsymbol{\lambda})\boldsymbol{V}^{-1} $$

	where $$$ \lambda $$$ is a vector containing all the eigenvalues and $$$ \boldsymbol{V} $$$ has the eigenvectors as columns.
	
*	Case of every real symmetric matrix can be decomposed into using only real-valued eigenvectors and eigenvalues:

	$$ \boldsymbol{A} = \boldsymbol{Q\Lambda Q^T} $$
	
	where $$$ \boldsymbol{Q} $$$ is an orthogonal matrix composed of eigenvectors of $$$ \boldsymbol{A} $$$, and $$$ \boldsymbol{\Lambda} $$$ is a diagonal matrix.
	
	
*	** Singular Value Decomposition **

	If a matrix is not square, the eigendecomposition is not defined, and we must use a singular value decomposition instead.



