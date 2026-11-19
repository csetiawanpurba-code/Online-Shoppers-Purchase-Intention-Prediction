# Online Shoppers Purchase Intention Prediction

## Project Overview

This project aims to predict whether an online shopping visitor will make a purchase based on their browsing behavior and session information.

Using the Online Shoppers Purchasing Intention Dataset, this project implements four supervised machine learning classification algorithms to predict customer purchase intention.

## Project Objectives

- Analyze online shopping visitor behavior.
- Identify patterns related to customer purchase decisions.
- Perform data preprocessing and feature transformation.
- Implement multiple machine learning classification algorithms.
- Evaluate and compare model performance.

## Dataset

Dataset: Online Shoppers Purchasing Intention Dataset

Target Variable: Revenue

The Revenue variable indicates whether a visitor completed a purchase.

The dataset contains numerical and categorical features, including:

- Administrative
- Administrative_Duration
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- SpecialDay
- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType
- Weekend

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Machine Learning Algorithms

### 1. Logistic Regression

A supervised classification algorithm used to estimate the probability of a visitor making a purchase.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to perform classification.

### 3. K-Nearest Neighbors (KNN)

A distance-based classification algorithm that predicts the class of a visitor based on its nearest neighbors.

### 4. Support Vector Machine (SVM)

A supervised learning algorithm that identifies an optimal decision boundary to classify visitors into purchase and non-purchase categories.

## Methodology

1. Load the dataset using Pandas.
2. Separate input features and target variables.
3. Identify numerical and categorical features.
4. Handle missing values using SimpleImputer.
5. Transform categorical features using OneHotEncoder.
6. Build preprocessing and classification pipelines.
7. Split the dataset into training and testing sets (80:20).
8. Train the machine learning models.
9. Generate predictions.
10. Evaluate model performance.

## Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These evaluation metrics help assess the classification performance of each algorithm.


## Installation and Usage

### Clone the Repository


git clone https://github.com/csetiawanpurba-code/Online-Shoppers-Purchase-Intention-Prediction.git


### Navigate to the Project Directory


cd Online-Shoppers-Purchase-Intention-Prediction


### Install Dependencies


pip install -r requirements.txt


### Run the Models


python LogisticRegression.py
python RandomForest.py
python knn.py
python svm.py


## Author

Christy Setiawan Purba

GitHub: https://github.com/csetiawanpurba-code
