# Week 1

## Neural Network Intuition

Neural networks, originally designed to mimic the brain’s learning process, have evolved into a powerful engineering tool with little resemblance to biological neurons. After cycles of popularity since the 1950s, they surged post-2005 as "deep learning," driven by big data and GPUs. Breakthroughs in speech recognition, computer vision (like ImageNet 2012), and NLP followed. While artificial neurons loosely mirror biological ones (processing inputs into outputs), modern advancements prioritize computational efficiency over brain-inspired designs. Today, neural networks thrive by leveraging massive datasets and hardware, not biological fidelity.

![image](https://github.com/user-attachments/assets/4bec0064-ba66-4893-993f-bd48f87830ef)
![image](https://github.com/user-attachments/assets/2048a32f-0886-48be-870e-53bef6feacc9)
![image](https://github.com/user-attachments/assets/531a2032-4a74-4ed0-8164-393005343420)
![image](https://github.com/user-attachments/assets/f147401e-0663-4694-a835-e0618ed20d59)
### Face Recognition Using Neural Network
![image](https://github.com/user-attachments/assets/dc3072bc-8723-44d6-9777-32be8ae29470)
### Car Classification Using Neural Network
![image](https://github.com/user-attachments/assets/b24d2aa2-8dc6-4da3-87fa-2d193ac34e9e)
### Tensorflow Implementation
![image](https://github.com/user-attachments/assets/bfde3355-dbf5-481f-801e-7045672d6256)
### Data in TensorFlow
![image](https://github.com/user-attachments/assets/a4f15aff-2598-4368-a6ab-6084c203e5ca)
## Building a neural network
In TensorFlow, building a neural network is streamlined using the **Sequential API**, which automatically chains layers (e.g., `Dense` with `sigmoid` activation). After defining the architecture (e.g., `model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, activation='sigmoid')])`), training involves two key steps: configuring the model with `compile()` (e.g., `model.compile(optimizer='sgd', loss='binary_crossentropy')`) and fitting it to data `(X, Y)` using `fit()` (e.g., `model.fit(X, Y, epochs=100)`). For inference, `predict()` handles forward propagation on new inputs (e.g., `predictions = model.predict(X_new)`). By convention, layers are nested directly in `Sequential()`, reducing code verbosity. While libraries abstract away manual implementation, understanding underlying mechanics, like forward propagation, remains crucial for troubleshooting and customization. This balance (leveraging frameworks for efficiency while grasping core concepts) empowers practitioners to deploy and debug neural networks confidently.  
![image](https://github.com/user-attachments/assets/fa0052d7-29b1-4b58-9a71-16ecb076dc3c)
### Digit Classification Model
![image](https://github.com/user-attachments/assets/32f025c5-9d19-4b52-8183-161911d16bbb)
### Forward propagation in a single layer

To implement forward propagation manually, we first initialize weight matrices and bias vectors. We then compute layer outputs through sequential matrix operations: multiplying inputs by weights, adding biases, and applying activation functions. These NumPy operations demonstrate how values propagate through the network. While this reveals the core mathematics behind neural networks, we typically rely on frameworks like TensorFlow for efficient, large-scale implementations. The exercise helps us understand the fundamental computations that power deep learning systems.
![image](https://github.com/user-attachments/assets/08048491-9489-4abf-b8a1-0d28b3dfaeb7)

### General implementation of forward propagation
We can generalize the forward propagation implementation by creating a reusable dense layer function. This function takes the previous layer's activations, weight matrix W (where columns represent individual neuron weights), and bias vector b as inputs. The implementation first determines the number of units in the layer from W's shape, initializes an output activation array, then iteratively computes each neuron's activation through vectorized operations. For each neuron, we extract its weights from W's columns, compute z = w·a_prev + b, and apply the sigmoid activation. By sequentially applying this dense function - first with input features X, then with each subsequent layer's outputs, we build complete forward propagation through the network. This modular approach avoids hardcoding per-neuron calculations while maintaining clarity about the underlying mathematics. Understanding this implementation helps debug neural networks even when using frameworks like TensorFlow, as it reveals the core computations these libraries optimize. The knowledge proves particularly valuable when diagnosing unexpected behavior or performance issues in practical applications.
![image](https://github.com/user-attachments/assets/d530b87d-09c8-414c-849d-c6aed6813752)
### How neural networks are implemented efficiently

The scalability of modern neural networks stems from their vectorized implementation using matrix operations. Researchers leverage parallel computing hardware like GPUs that excel at large matrix multiplications. While a naive implementation might use explicit for-loops to compute each neuron's activation separately, the vectorized approach replaces this with efficient matrix operations. By representing weights W as a matrix (where columns correspond to individual neurons' weights) and inputs X as a 2D array, forward propagation simplifies to Z = matmul(X, W) + B, followed by element-wise activation. This matmul operation performs the equivalent of all individual neuron calculations simultaneously. The vectorized implementation treats all quantities (inputs, weights, biases, and outputs) as 2D arrays/matrices, enabling hardware-accelerated parallel computation. This matrix-based approach is fundamental to training large neural networks efficiently and is why deep learning frameworks heavily optimize these linear algebra operations. Understanding this vectorization principle helps appreciate why neural networks can scale while maintaining computational efficiency.
![image](https://github.com/user-attachments/assets/d7b7ed75-c86d-4bc9-b82f-56edbcf282ad)

### Matrix multiplication
Matrix multiplication is fundamentally built upon vector dot products, systematically extended to higher dimensions. The operation between two matrices computes each element of the output as the dot product of the corresponding rows from the first matrix and columns from the second matrix. 

Key insights about matrix multiplication:
1. **Vector-Vector Case**: The dot product of vectors a and w (a·w) equals aᵀw when treating vectors as column matrices
2. **Vector-Matrix Case**: Multiplying a row vector aᵀ by matrix W computes dot products with each column of W
3. **Matrix-Matrix Case**: For AᵀW, each element (i,j) results from the dot product of row i of Aᵀ with column j of W

The transpose operation converts columns to rows, enabling proper dimensional alignment for multiplication. This systematic approach allows neural networks to compute all layer activations simultaneously through batched matrix operations, rather than processing neurons individually. The matmul operation in NumPy/TensorFlow implements precisely this pattern of grouped dot products, providing the computational efficiency essential for deep learning.
![image](https://github.com/user-attachments/assets/3ab813dc-a21a-4056-960a-785e844a1594)
![image](https://github.com/user-attachments/assets/ba076ee4-96d1-4d6a-a11c-97c90ad1177b)

### Matrix Multiplication Fundamentals

The general matrix multiplication operation follows these key principles:

1. **Dimensional Requirements**:
   - For matrix A (m×n) and W (n×p), multiplication is valid only when A's column count matches W's row count
   - Resulting matrix Z (m×p) preserves A's row dimension and W's column dimension

2. **Computation Process**:
   - Each element zᵢⱼ is computed as the dot product of row i from Aᵀ with column j from W
   - Visual grouping (shaded rows/columns) helps track which vector combinations produce each output element

3. **Neural Network Implementation**:
   - Layer activations become matrix rows (after transpose)
   - Weight matrices organize neuron parameters column-wise
   - Enables simultaneous computation of all neurons' outputs via matmul

4. **Performance Impact**:
   - Vectorized implementations leverage parallel hardware
   - Eliminates explicit neuron-by-neuron loops
   - Provides orders-of-magnitude speedup over naive implementations

5. **Implementation Insight**:
   - AᵀW computation mirrors the neural network forward pass
   - Each matmul operation computes a full layer's activations
   - Proper dimensional alignment ensures correct neuron connections

This matrix approach transforms O(n) sequential operations into O(1) parallelized computations, enabling modern deep learning scalability. The dimensional constraints naturally enforce proper network architecture connectivity.
![image](https://github.com/user-attachments/assets/9713802c-ad51-495f-83fa-2c435cd50db5)

### Matrix multiplication code
Vectorized forward prop uses `Z = matmul(Aᵀ,W) + B` to compute all neuron activations simultaneously via matrix ops, where `Aᵀ` is input, `W` contains weights (columns=neurons), and `B` is bias. The output `A_out = g(Z)` applies activation element-wise. This approach leverages parallel hardware for efficient neural network computation.
![image](https://github.com/user-attachments/assets/4bd37201-709c-49e6-9fd1-c6e43a67026d)
# Week 2











