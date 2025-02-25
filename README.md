# Active Learning-Based Comparative Analysis of Machine Learning Models for Weather Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Implementation Overview](#implementation-overview)
3. [Dataset Overview](#dataset-overview)
4. [Model Architecture](#model-architecture)
5. [Active Learning Strategy](#active-learning-strategy)
6. [Simulation Algorithm](#simulation-algorithm)
7. [Performance Metrics](#performance-metrics)
8. [Simulation Analysis](#simulation-analysis)
9. [Conclusion](#conclusion)

---

## <a id="introduction"></a>Introduction
This project focuses on **weather prediction** using two machine learning models: **Random Forest Classifier** and **Logistic Regression**. The key highlight of this project is the integration of **Active Learning**, which iteratively improves model performance by selectively adding the most uncertain samples to the training set. The goal is to compare the performance of the two models and analyze the impact of active learning on their accuracy and generalization.

---

## <a id="implementation-overview"></a>Implementation Overview
The project follows a structured pipeline:
1. **Dataset Preprocessing**:
   - Load and clean the dataset.
   - Handle missing values and encode categorical variables.
2. **Feature Engineering**:
   - Scale numerical features.
   - Transform categorical features for model input.
3. **Model Training**:
   - Train **Random Forest Classifier** and **Logistic Regression** models.
4. **Active Learning**:
   - Iteratively update the training dataset by selecting the most uncertain predictions based on entropy.
5. **Model Evaluation**:
   - Analyze models using accuracy, precision, recall, F1-score, and confusion matrices.
6. **Visualization**:
   - Plot validation accuracy and confusion matrices to compare model performance.

---

## <a id="dataset-overview"></a>Dataset Overview
The dataset consists of meteorological parameters such as:
- **Features**: Temperature, humidity, wind speed, precipitation, etc.
- **Target Variable**: Binary classification (Rain or No Rain).

The dataset is split into training and testing sets, with 90% of the data used for active learning and 10% for validation.

---

## <a id="model-architecture"></a>Model Architecture
### 1. Random Forest Classifier
- An ensemble model that builds multiple decision trees and aggregates their results.
- Parameters:
  - `n_estimators`: Number of trees in the forest.
  - `warm_start`: Allows incremental training.

### 2. Logistic Regression
- A linear model for binary classification.
- Parameters:
  - `max_iter`: Maximum number of iterations for convergence.
  - `warm_start`: Allows incremental training.

---

## <a id="active-learning-strategy"></a>Active Learning Strategy
Active learning is implemented to improve model performance by focusing on uncertain predictions. The steps are:
1. Train the initial model on a small labeled dataset.
2. Use the model to predict probabilities for the unlabeled pool.
3. Compute entropy to measure uncertainty.
4. Select the most uncertain samples and add them to the training set.
5. Retrain the model with the updated dataset.
6. Repeat the process for a fixed number of epochs.

---

## <a id="simulation-algorithm"></a>Simulation Algorithm
The simulation algorithm for active learning-based weather prediction is as follows:

1. **Initialize**:
   - Split the dataset into a small labeled set (`dataset_X`, `dataset_y`) and a large unlabeled pool (`unlabeled_pool_X`, `unlabeled_pool_y`).
   - Initialize the machine learning model (Random Forest or Logistic Regression).

2. **Train Initial Model**:
   - Train the model on the initial labeled dataset.

3. **Active Learning Loop**:
   - For each epoch:<br>
     a. Use the model to predict probabilities for the unlabeled pool.<br>
     b. Compute entropy for each prediction to measure uncertainty.<br>
     c. Identify the most uncertain sample (highest entropy).<br>
     d. Add the most uncertain sample to the labeled dataset.<br>
     e. Retrain the model on the updated labeled dataset.<br>
     f. Evaluate the model on the validation set and record performance metrics.<br>

4. **Termination**:
   - Stop after a fixed number of epochs or when the model's performance stabilizes.

5. **Final Evaluation**:
   - Evaluate the final model on the test set using accuracy, precision, recall, and F1-score.

---

## <a id="performance-metrics"></a>Performance Metrics
The models are evaluated using the following metrics:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of true positive predictions.
- **Recall**: Proportion of actual positives correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall.

---

## <a id="simulation-analysis"></a>Simulation Analysis
### Validation Accuracy Plot
- The validation accuracy of both models improves over epochs as uncertain samples are added to the training set.
- Random Forest achieves higher accuracy compared to Logistic Regression.

### Confusion Matrix
- **Random Forest**: Fewer false negatives, indicating better detection of rainy days.
- **Logistic Regression**: Higher misclassification rate, especially for rainy days.

---

## <a id="conclusion"></a>Conclusion
- **Random Forest Classifier** outperforms **Logistic Regression** in weather prediction, achieving higher accuracy, precision, recall, and F1-score.
- The integration of **Active Learning** significantly improves model performance by dynamically selecting the most informative training samples.
- This approach ensures better generalization, especially for uncertain cases, making it suitable for real-world weather forecasting applications.

---
