# DataScience_DeepLearning----ANN-ModelBuilding----Bankrupt_Prediction_ANN
# Corporate Bankruptcy Prediction using Artificial Neural Networks (ANN)

An end-to-end Deep Learning classification project designed to predict corporate bankruptcy based on historical financial performance indicators. This repository features a complete data science pipeline implementing a sequential Artificial Neural Network (ANN) in Keras/TensorFlow, achieving a benchmark classification accuracy of **96%**.

## 🚀 Project Overview
Predicting corporate financial distress is a critical risk-management task that requires modeling complex, non-linear relationships across highly multi-dimensional data boundaries. This project establishes a streamlined deep learning workflow to ingest corporate metrics, handle feature variances via standard scaling, and optimize a dense neural network sequence designed to maximize model generalization and predictive reliability.

## 📂 Repository Structure
* **`python.ipynb`**: The master Jupyter Notebook containing exploratory data validation, structural feature preprocessing, neural network model compilation, and evaluation metrics.
* **`data.csv`**: The operational dataset containing financial balance metrics, liquidity indicators, growth rates, and debt ratios.

## 📊 Pipeline & Preprocessing Workflow
1. **Data Ingestion**: Parses extensive financial records from `data.csv`, focusing on key operational factors (such as Return on Assets, asset turnovers, quick ratios, and cash flow margins).
2. **Feature Segregation**: Isolates the independent financial metric space from the categorical bankruptcy target flag (`Bankrupt?`).
3. **Feature Scaling**: Applies standard scaling transformations to balance feature weightings, ensuring stable gradient descent updates during neural network training.
4. **Data Partitioning**: Standardizes feature sets into separate training and validation partitions to monitor model convergence and guard against overfitting.

## 🧠 Neural Network Architecture
The baseline architecture features a dense, multi-layer sequential topology implemented in Keras/TensorFlow:
* **Input & Hidden Layer 1**: Fully connected layer utilizing Rectified Linear Unit (`ReLU`) activations to map multi-dimensional financial weights.
* **Hidden Layer 2**: Dense hidden dimension stacks to refine continuous patterns across deeply learned decision nodes.
* **Output Node**: Configured with a single sigmoid prediction neuron to output discrete bankruptcy classification probabilities.
* **Compilation**: Optimized using the **Adam** gradient descent variant and evaluated using classification error loss objectives.

## 📈 Model Performance & Validation
The trained Artificial Neural Network classifier achieves a high performance benchmark:
* **Overall Model Accuracy:** **96%**
* **Statistical Performance Diagnostics**: Model metrics are verified using standard Scikit-Learn evaluation arrays. A complete **Classification Report** is compiled to trace precision distributions, recall thresholds, and F1-score balances, confirming robust performance across targeted class boundaries.

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python
* **Deep Learning Framework:** Keras / TensorFlow
* **Machine Learning & Preprocessing:** Scikit-Learn
* **Data Wrangling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
