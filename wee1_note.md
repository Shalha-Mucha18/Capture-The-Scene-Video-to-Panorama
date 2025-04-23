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


---

## Unsupervised Learning

**Unsupervised Learning** deals with **unlabeled data**. The model explores the underlying structure or patterns in the data without predefined labels.

### Main Applications

- **Clustering** – Grouping similar data points (e.g., customer segmentation)
- **Dimensionality Reduction** – Reducing the number of features (e.g., PCA)
- **Anomaly Detection** – Identifying outliers or unusual data patterns

![image](https://github.com/user-attachments/assets/b214869e-a929-416d-8e84-317ae093a17c)


### Linear Regression Model

Linear Regression is a supervised learning algorithm used to predict a **continuous output variable** based on one or more input features.  
It models the relationship between the input variables (**X**) and the output variable (**y**) using a **straight line**.

![image](https://github.com/user-attachments/assets/0e09ee51-7e61-4117-97db-38e7f48e9e2f) 
![image](https://github.com/user-attachments/assets/81cf56e3-115e-4adc-bb98-1adbb988f334)




### Cost Function

The **cost function** calculates the difference between the predicted values and the actual values from the dataset.

- It quantifies the error in prediction.
**Goal:** Minimize the cost function to improve model accuracy.

  ![image](https://github.com/user-attachments/assets/621f1307-9a16-49bc-8842-6afe7af8c599)


### Gradient Descent

**Gradient Descent** is an optimization algorithm used to minimize the cost function (or loss function) in machine learning models.

- It works by **iteratively updating the model parameters** in the direction that reduces the cost.
- Helps the model learn optimal values for better predictions.
![image](https://github.com/user-attachments/assets/6c49fb31-75e6-44f9-974b-8e7da2aeaf3a)
![image](https://github.com/user-attachments/assets/2caf6571-8b78-46cd-9b57-a7404500129f)

#### Gradient descent for linear regression
![image](https://github.com/user-attachments/assets/6c90aa0b-02fa-418a-b9e0-caeb5f0cccda)


### Learning Rate

The **learning rate**, often denoted by **α (alpha)**, is a **hyperparameter** that determines the size of the steps taken during gradient descent when updating model parameters.

- A **small α** results in slow learning and might take longer to converge.
- A **large α** might overshoot the optimal value or even diverge.

  ![image](https://github.com/user-attachments/assets/53d4a954-4656-40f7-89d3-f9f620e54548)


---


Unsupervised learning deals with **unlabeled data**; the model explores data structure or patterns without predefined labels.

**Main applications:**
- Clustering
- Dimensionality reduction
- Anomaly detection
