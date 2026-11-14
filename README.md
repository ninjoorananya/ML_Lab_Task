# 🤖 ML — Machine Learning Lab

This repository contains two machine learning experiments implemented in Python:

1. **Random Forest Classifier** — Breast Cancer (Wisconsin) Dataset
2. **Backpropagation Neural Network** — XOR Problem & Iris Dataset

---

## 📁 Project Structure

```
ml/
├── README.md
├── RandomForest/
│   ├── data.csv                   # Breast Cancer Wisconsin dataset
│   ├── random_forest.py           # Main Python script
│   ├── randomforestlab.ipynb      # Jupyter Notebook version
│   └── true_vs_predicted.png      # True vs Predicted plot
│
└── Backpropagation/
    ├── Iris (1).csv               # Iris dataset
    ├── backpropagation.py         # Main Python script
    └── backpropagationlab.ipynb   # Jupyter Notebook version
```

---

## 🌲 Random Forest Classifier

**Dataset:** Breast Cancer Wisconsin (`data.csv`)  
**Task:** Binary classification — Benign vs Malignant tumors  
**Model:** `RandomForestClassifier` (scikit-learn)

### How It Works
- Loads and explores the dataset
- Drops non-feature columns (`id`, `diagnosis`, `Unnamed: 32`)
- Splits data into 80% train / 20% test
- Trains a Random Forest with 3 estimators and 5 max features
- Evaluates using Accuracy, Confusion Matrix, and Classification Report
- Plots True vs Predicted values

### 📊 Output Screenshot

![Random Forest Output](rf_output.png)

### 📈 True vs Predicted Plot

![True vs Predicted](RandomForest/true_vs_predicted.png)

### ✅ Results Summary

| Metric              | Value        |
|---------------------|-------------|
| Training Accuracy   | 98.24%      |
| Testing Accuracy    | 95.61%      |
| Correct Predictions | 109 / 114   |
| Misclassified       | 5 / 114     |

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Benign (B)     | 0.96      | 0.97   | 0.97     | 71      |
| Malignant (M)  | 0.95      | 0.93   | 0.94     | 43      |
| **Accuracy**   |           |        | **0.96** | 114     |

---

## 🧠 Backpropagation Neural Network

**Datasets:** XOR logic gate + Iris flower dataset (`Iris (1).csv`)  
**Task:** XOR binary classification & Iris 3-class classification  
**Model:** Custom neural network with sigmoid/softmax activations

### How It Works

#### Part 1 — XOR Problem
- 2-input, 3-hidden, 1-output neuron network
- Trains for 500 epochs using gradient descent
- Demonstrates that a neural network can learn non-linear XOR logic

#### Part 2 — Iris Dataset
- 4-input, 8-hidden, 3-output neuron network
- Features normalized using z-score standardization
- One-hot encoded target labels
- Softmax output layer for multi-class classification
- Trains for 5000 epochs

### 📊 Output Screenshot

![Backpropagation Output](bp_output.png)

### ✅ Results Summary

**XOR Problem:**

| Metric              | Value   |
|---------------------|---------|
| Correct Predictions | 4 / 4   |
| Accuracy            | 100.00% |

**Iris Dataset:**

| Metric              | Value       |
|---------------------|-------------|
| Training Accuracy   | 98.00%      |
| Correct Predictions | 147 / 150   |
| Misclassified       | 3 / 150     |

| Class           | Accuracy | Correct |
|-----------------|----------|---------|
| Iris-setosa     | 100.00%  | 50/50   |
| Iris-versicolor | 96.00%   | 48/50   |
| Iris-virginica  | 98.00%   | 49/50   |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Random Forest
```bash
cd RandomForest
python random_forest.py
```

### Backpropagation
```bash
cd Backpropagation
python backpropagation.py
```

### Jupyter Notebooks
Open in VS Code or Jupyter:
- `RandomForest/randomforestlab.ipynb`
- `Backpropagation/backpropagationlab.ipynb`

---

## 🛠️ Tech Stack

| Library      | Version  | Purpose                        |
|-------------|----------|-------------------------------|
| Python       | 3.15     | Programming language           |
| NumPy        | 2.5.4    | Numerical computations         |
| Pandas       | 3.1.1    | Data loading & manipulation    |
| scikit-learn | 1.9.1    | Random Forest & metrics        |
| Matplotlib   | 3.11.2   | Visualization                  |
