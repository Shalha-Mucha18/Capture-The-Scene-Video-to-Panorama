# Week 1
## Operations with vectors:
Vectors are objects that can represent movement in space or describe an object's attributes. In physics, vectors are often thought of as entities that move us through physical space, while in data science, they are used to represent attributes of objects, like a house's characteristics (e.g., square footage, number of bedrooms, price).

The key operations on vectors include:

1. **Addition**: Vectors can be added together. The order of addition doesn’t affect the result (commutative property), and the addition of three vectors can be grouped in any way (associative property).
2. **Scalar Multiplication**: A vector can be scaled by multiplying it by a number (scalar), which changes its magnitude.
3. **Vector Subtraction**: Subtracting a vector is equivalent to adding the negative of that vector.
4. **Coordinate Systems**: Vectors are often represented in a coordinate system using basis vectors (e.g., i and j) to describe directions.

These operations allow vectors to be used not only in geometric spaces but also in non-spatial contexts, like data science, where vectors can represent complex attributes of objects.

### Mathematical Equation:

For **vector addition** and **scalar multiplication**:

Let:

* $\mathbf{r} = \langle 3, 2 \rangle$,
* $\mathbf{s} = \langle -1, 2 \rangle$.

The **addition** of $\mathbf{r}$ and $\mathbf{s}$ is:

$$
\mathbf{r} + \mathbf{s} = \langle 3 + (-1), 2 + 2 \rangle = \langle 2, 4 \rangle
$$

Scalar multiplication of $\mathbf{r}$ by 2:

$$
2\mathbf{r} = 2 \times \langle 3, 2 \rangle = \langle 6, 4 \rangle
$$

These equations demonstrate how vector addition and scalar multiplication work in a 2D space.
# Week 2
## Length (or Magnitude) of a Vector:

The length or magnitude of a vector is calculated using the Pythagorean theorem. For a 2D vector 
$\mathbf{r} = a\mathbf{i} + b\mathbf{j}$, the length of the vector is the square root of the sum of the squares of its components:

$$
\text{Length of } \mathbf{r} = \sqrt{a^2 + b^2}
$$

This idea can be generalized to vectors with more dimensions or even those with different units (like length, time, price, etc.).

## Dot Product:

The dot product (or inner product) of two vectors $\mathbf{r}$ and $\mathbf{s}$ is calculated by multiplying their corresponding components and summing the results:

$$
\mathbf{r} \cdot \mathbf{s} = r_i \times s_i + r_j \times s_j
$$

This results in a scalar value. The dot product has several important properties:

- **Commutativity**: The dot product is commutative, meaning 
  $$
  \mathbf{r} \cdot \mathbf{s} = \mathbf{s} \cdot \mathbf{r}
  $$

- **Distributivity**: The dot product is distributive over vector addition:
  $$
  \mathbf{r} \cdot (\mathbf{s} + \mathbf{t}) = \mathbf{r} \cdot \mathbf{s} + \mathbf{r} \cdot \mathbf{t}
  $$

- **Associativity with scalar multiplication**:
  $$
  a (\mathbf{r} \cdot \mathbf{s}) = \mathbf{r} \cdot (a \mathbf{s})
  $$

## Connection Between Dot Product and Length:

The dot product of a vector with itself gives the square of its length:

$$
\mathbf{r} \cdot \mathbf{r} = r_1^2 + r_2^2 + \dots + r_n^2
$$

This shows that the length of a vector can be found by taking the square root of the dot product of the vector with itself.

### Mathematical Equation:

The key equations are:

- **Length of a vector** $\mathbf{r}$:
  $$
  \text{Length of } \mathbf{r} = \sqrt{r_1^2 + r_2^2 + \dots + r_n^2}
  $$

- **Dot product of vectors** $\mathbf{r}$ and $\mathbf{s}$:
  $$
  \mathbf{r} \cdot \mathbf{s} = r_1 s_1 + r_2 s_2 + \dots + r_n s_n
  $$

- **Length from dot product**:

  $$
  \text{Length of } \mathbf{r} = \sqrt{\mathbf{r} \cdot \mathbf{r}}
  $$

These equations define how to compute both the magnitude (length) and the dot product of vectors, essential tools in vector analysis.



  $$
  \text{Length of } \mathbf{r} = \sqrt{\mathbf{r} \cdot \mathbf{r}}
  $$

These equations define how to compute both the magnitude (length) and the dot product of vectors, essential tools in vector analysis.

