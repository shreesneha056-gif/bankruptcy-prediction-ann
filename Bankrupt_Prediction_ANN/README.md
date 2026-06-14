# Flight Class & Pattern Classification using Artificial Neural Networks (ANN)

An end-to-end Deep Learning classification project designed to process structured aviation features and predict targeted operational categories. This repository features a comprehensive data science pipeline implementing a sequential Artificial Neural Network (ANN) in Keras/TensorFlow, achieving a benchmark classification accuracy of **96%**.

## 🚀 Project Overview
Predicting complex target boundaries within structured aviation datasets involves managing sparse categorical distributions, high feature multi-collinearity, and non-linear interactions. This project establishes a complete workflow to handle categorical transformations, balance feature weights via standardization, and optimize a deep neural network sequence designed to maximize model generalization and classification performance.

## 📂 Repository Structure
* **`python.ipynb`**: The master Jupyter Notebook containing the full data pipeline, including data exploration, structural preprocessing engineering, model training loops, and evaluation metrics.

## 📊 Core Pipeline Workflow

### 1. Data Cleaning & Feature Preprocessing
* **Feature Processing**: Raw categorical parameters and metadata flags are transformed into sparse tracking representations optimized for neural networks.
* **Feature Scaling**: Implements standard scaler transformations across the input layer to prevent feature dominance and ensure stable, balanced gradient updates during optimization.

### 2. Deep Learning Architecture
The neural network features a dense, multi-layer sequential layout:
* **Input / Hidden Layer 1**: Fully connected layer utilizing Rectified Linear Unit (`ReLU`) activation functions to model complex, non-linear relationships.
* **Hidden Layer 2**: Dense dimension reductions to refine feature mapping patterns across deeply learned decision spaces.
* **Output Layer**: Configured with a dedicated probability activation tracking block mapping binary or multi-class parameters.
* **Compilation**: Optimized using the **Adam** gradient descent variant and evaluated using classification error loss objectives.

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python
* **Deep Learning Framework:** Keras / TensorFlow
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn

## 📈 Model Performance & Evaluation
The final trained classifier is comprehensively audited across independent test partitions using Scikit-Learn's standard validation report matrix:
* **Overall Model Accuracy:** **96%**
* **Diagnostics**: Performance metrics are parsed through a complete **Classification Report**, confirming robust precision, recall alignments, and strong F1-scores across all targeted data boundaries.

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME