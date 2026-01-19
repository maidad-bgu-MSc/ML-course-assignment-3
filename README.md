# Assignment 3: Multi-Layer Neural Network Analysis

This repository contains the implementation and analysis for **Assignment 3**, focusing on extending a NumPy-based Multi-Layer Perceptron (MLP) and comparing its performance with modern deep learning frameworks.

## Authors
* **Maidad Maissel** - *Ben-Gurion University of the Negev*
* **Yuval Cohen**

## Project Overview
The primary goal of this assignment was to transition from a basic single-hidden-layer architecture to a deeper network using the MNIST handwritten digit dataset. This project involves:
1.  **Manual Implementation:** Extending the scratch MLP code from Chapter 11 of *Machine Learning with PyTorch and Scikit-Learn* (Raschka et al.) to support two hidden layers.
2.  **Framework Comparison:** Implementing an equivalent architecture in PyTorch to benchmark performance.
3.  **Evaluation:** Using Macro-averaged AUC (One-vs-Rest) and Accuracy to assess model robustness across all 10 digit classes.



## Methodology
The scratch implementation was extended from a $D \rightarrow H_1 \rightarrow O$ structure to a $D \rightarrow H_1 \rightarrow H_2 \rightarrow O$ structure.
* **Input Layer:** 784 units (MNIST images normalized to the range $[-1, 1]$).
* **Hidden Layer 1:** 500 units.
* **Hidden Layer 2:** 250 units (added for this assignment).
* **Output Layer:** 10 units (Softmax).
* **Preprocessing:** A stratified split of 70% training and 30% testing was applied ($49,000$ training samples, $21,000$ testing samples).
* **Optimization:** Stochastic Gradient Descent (SGD) with a learning rate of 0.1 and a minibatch size of 100.

## Performance Results
The models were trained for 20 epochs. The results demonstrate that the deeper architecture significantly improved the model's capacity to capture non-linear features.

| Model Architecture | Final MSE | Accuracy (%) | Macro AUC |
| :--- | :--- | :--- | :--- |
| Custom NumPy (1-Hidden) | 0.02 | 89.37% | 0.990 |
| **Custom NumPy (2-Hidden)** | **0.01** | **92.68%** | **0.993** |
| PyTorch (1-Hidden) | - | 89.89% | 0.9899 |

### Key Observations
* **Impact of Depth:** Adding a second hidden layer to the manual implementation increased accuracy by **3.31%**.
* **Numerical Stability:** The implementation utilizes stabilized Softmax and Sigmoid functions to prevent overflow/underflow during exponential calculations.
* **Metric Robustness:** The high Macro AUC suggests the model is extremely robust at distinguishing between different digits across all classes.

## Repository Structure
* `ch11.ipynb`: The primary Jupyter Notebook containing the NumPy implementation, PyTorch comparison, and evaluation plots.
* `Report.pdf`: A detailed PDF report explaining the methodology, mathematical derivations, and final results.
* `README.md`: Project documentation and summary.

## Installation & Requirements
To run the notebook locally, ensure you have the following dependencies installed:
```bash
pip install numpy matplotlib scikit-learn torch
