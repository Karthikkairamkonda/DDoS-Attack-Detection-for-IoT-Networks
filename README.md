🚀 DDoS Attack Detection in IoT Networks

🔍 Project Overview

This project implements a machine learning pipeline for detecting Distributed Denial-of-Service (DDoS) attacks in IoT networks. It utilizes multiple classification models, data preprocessing techniques, feature engineering, and performance evaluation methods to classify network traffic anomalies effectively.

📊 Dataset

The dataset used for this project consists of multiple network traffic logs, including:

DrDOS_DNS.csv

DrDOS_LDAP.csv

Syn.csv

These datasets contain labeled network traffic data, allowing for supervised learning.

📌 Key Features

📂 Load and preprocess network traffic datasets.

⚡ Handle missing values, encode categorical data, and normalize features.

🧠 Train multiple machine learning models:

🤖 Naive Bayes

🌲 Random Forest

⚡ Support Vector Machine (SVM)

🚀 XGBoost

🔥 AdaBoost

👥 K-Nearest Neighbors (KNN)

📊 Evaluate model performance using key metrics.

🖥️ Visualize confusion matrix for insights.

🛠 Requirements

Ensure you have the following dependencies installed:

🚀 How to Run

Clone the repository:

Run the script:

Use the GUI to:

📂 Load the dataset

⚙️ Preprocess the data

🤖 Train a selected model

📊 View performance metrics and insights

📈 Data Preprocessing

✅ Handle missing values by replacing them with appropriate statistics.

🏷️ Label encoding for categorical columns.

🔄 Normalization applied to feature columns.

✂️ Data split into training and testing sets (80-20 ratio).

🔬 Feature Engineering

📊 Identified most important features using Random Forest Feature Importance.

🔍 Detected and handled outliers using boxplots.

🚀 Improved model efficiency by selecting top relevant features.

🏆 Machine Learning Models and Results

Three machine learning models were trained with hyperparameter tuning:

1️⃣ Support Vector Machine (SVM)

Hyperparameters Tuned:

C (Regularization)

Kernel Type (linear, rbf)

Performance:

Achieved strong precision, recall, and F1 scores.

Competitive accuracy for DDoS detection.

2️⃣ Random Forest Classifier

Hyperparameters Tuned:

n_estimators (Number of trees)

max_depth (Tree depth)

Performance:

High accuracy with balanced classification results.

Robust feature importance analysis.

3️⃣ XGBoost Classifier

Hyperparameters Tuned:

learning_rate

n_estimators

Performance:

Achieved high precision and recall.

Effective in detecting DDoS attacks.

📌 Performance Comparison:

Consolidated model performance using accuracy, precision, recall, F1-score, and confusion matrices.

📊 Results and Findings

✅ Machine Learning Models: Random Forest and XGBoost performed exceptionally well in identifying DDoS traffic.

🏆 Feature Engineering: Selected key features significantly boosted model accuracy.

📉 EDA Insights: Identified class imbalances and handled missing values.

🔥 Confusion Matrix Analysis: Showcased model performance visually.

📌 Usage Guide

Clone the Repository

Create and Activate a Virtual Environment

For Windows:

For macOS/Linux:

Install Dependencies

Run the Application

Outputs Generated:

📊 EDA visualizations and reports.

🔍 Feature importance plots.

🏆 Model performance metrics and confusion matrices.
