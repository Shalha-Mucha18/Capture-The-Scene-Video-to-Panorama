# Machine Learning

Machine learning is a field of artificial intelligence that teaches computers to learn from data without being explicitly programmed.

## Types of Machine Learning

- **Supervised Learning**
- **Unsupervised Learning**
- **Semi-Supervised Learning**
- **Reinforcement Learning**

---

## Supervised Learning

In supervised learning, the algorithm learns from labeled data, meaning each training example has an input and a known correct output (label). The model makes predictions and adjusts based on the accuracy of those predictions.

### Types of Supervised Learning Tasks

| Task Type      | Output Type          | Example                             |
|----------------|----------------------|-------------------------------------|
| Regression     | Continuous Value     | Predicting house prices, stock value |
| Classification | Discrete Class Label | Spam detection                      |

### Linear Regression Model

Linear Regression is a supervised learning algorithm used to predict a **continuous output variable** based on one or more input features.  
It models the relationship between the input variables (**X**) and the output variable (**y**) using a **straight line**.

### Cost Function

The **cost function** calculates the difference between the predicted values and the actual values from the dataset.

- It quantifies the error in prediction.
**Goal:** Minimize the cost function to improve model accuracy.

### Gradient Descent

**Gradient Descent** is an optimization algorithm used to minimize the cost function (or loss function) in machine learning models.

- It works by **iteratively updating the model parameters** in the direction that reduces the cost.
- Helps the model learn optimal values for better predictions.

### Learning Rate

The **learning rate**, often denoted by **α (alpha)**, is a **hyperparameter** that determines the size of the steps taken during gradient descent when updating model parameters.

- A **small α** results in slow learning and might take longer to converge.
- A **large α** might overshoot the optimal value or even diverge.

---

## Unsupervised Learning

Unsupervised learning deals with **unlabeled data**; the model explores data structure or patterns without predefined labels.

**Main applications:**
- Clustering
- Dimensionality reduction
- Anomaly detection
