# Customer-Conversion-Prediction

Project Overview
The goal of the Customer Conversion Prediction project is to build a machine learning model that can predict whether a client will subscribe to insurance based on their demographic and marketing data. The project involves data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

Dataset
The dataset includes the following features:

age: Client's age (int64)
job: Type of job (object)
marital: Marital status (object)
education_qual: Education qualification (object)
call_type: Type of call (object)
day: Day of the month (int64)
mon: Month of the year (object)
dur: Duration of the call (int64)
num_calls: Number of calls made (int64)
prev_outcome: Outcome of the previous marketing campaign (object)
y: Target variable indicating if the client subscribed (Yes/No) (object)
Project Steps
1. Data Cleaning
Shape and Description: Understand the dataset's shape and summary statistics.
Outliers and Duplicates: Identify and handle outliers and duplicate records.
Null Values: Check and handle missing values.
2. Exploratory Data Analysis (EDA)
Categorical Features
Analyzed distribution for:

y (Target Variable)
job
marital
education_qual
call_type
mon
Numerical Features
Visualized distributions using seaborn for:
age
day
dur
num_calls
Visualized Customer Conversion by Day of the Month.
Performed bivariate analysis of several categorical columns.
Observed correlations using a heatmap.
3. Data Preprocessing
Encoding: Performed one-hot encoding on categorical features.
Resampling: Used ADASYN for data resampling to handle class imbalance.
Train-Test Split: Split the data into training and testing sets.
Scaling: Applied standard scaling to numerical features.
4. Model Building and Evaluation
Trained and evaluated the following models:

Logistic Regression
Random Forest Classifier
Decision Tree
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Naive Bayes
Gradient Boosting
Performed hyperparameter tuning for each model and identified the best parameters. Further evaluation was carried out using cross-validation and ROC AUC score for the top-performing models:

Random Forest Classifier
Decision Tree
KNN
Gradient Boosting
5. Model Results
Gradient Boosting emerged as the best model with a ROC AUC score of 0.986 and a mean accuracy of 0.773 (±0.119).
The most important feature for the prediction was duration.
6. Conclusion
The Gradient Boosting model is recommended for predicting client subscriptions to insurance. It demonstrates excellent prediction ability, high ROC AUC score, and consistent performance across various datasets. This model can help target potential customers accurately and save on marketing expenses.
