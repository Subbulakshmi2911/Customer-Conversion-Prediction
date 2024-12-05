Customer Conversion Prediction
Project Overview

The Customer Conversion Prediction project aims to build a machine learning model that predicts whether a client will subscribe to insurance based on demographic and marketing data. The project involves data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation. The goal is to create a model that helps businesses accurately identify potential clients, ultimately saving on marketing efforts and improving customer acquisition strategies.
Dataset

The dataset used in this project includes the following features:

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

    Shape and Description: The dataset was analyzed to understand its shape and summary statistics.
    Outliers and Duplicates: Identified and handled outliers and duplicate records to ensure data quality.
    Null Values: Checked and handled missing values appropriately, ensuring no loss of information.

2. Exploratory Data Analysis (EDA)
Categorical Features

    Target Variable (y): Analyzed the distribution of clients who subscribed vs. those who did not.
    Other Categorical Features: Examined distributions of job, marital, education_qual, call_type, and mon.

Numerical Features

    Analyzed the distributions of numerical features like age, day, dur, and num_calls.

3. Data Preprocessing

    Encoding: Applied Label Encoding to transform categorical features into numerical values for model compatibility.
    Resampling: Used SMOTENN (Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors) for handling class imbalance in the dataset.
    Train-Test Split: Split the dataset into training and testing sets to evaluate model performance.

4. Model Building and Evaluation

Several models were trained and evaluated to determine the best-performing one:

    Logistic Regression
    Random Forest Classifier
    Decision Tree
    Support Vector Machine (SVM)
    K-Nearest Neighbors (KNN)
    Gradient Boosting

5. Model Results

After evaluating multiple models, Gradient Boosting emerged as the best model with the following metrics:

    F1 Score: 0.587
    Mean Accuracy: 91%
    The most important feature for prediction was duration (the duration of the marketing call).

6. Conclusion

The Gradient Boosting model was identified as the most effective model for predicting customer subscription to insurance. It demonstrated a high ROC AUC score and consistent performance across different datasets, making it ideal for targeting potential clients and improving the efficiency of marketing campaigns.
