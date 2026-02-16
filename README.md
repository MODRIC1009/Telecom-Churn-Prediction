Telecom Customer Churn Prediction

Overview

This project builds an end-to-end machine learning pipeline to predict customer churn in a telecom company.
It includes exploratory data analysis, feature engineering, model training, evaluation, and deployment as a web application.

The goal is to identify customers at high risk of leaving so that the business can take proactive retention measures.

Problem Statement

Customer churn leads to significant revenue loss.
By predicting which customers are likely to churn, telecom companies can:

1. Improve retention strategies
2. Reduce customer acquisition costs
3. Increase lifetime value

Dataset

Telco Customer Churn Dataset

Contains customer information such as:

1. Demographics
2. Account details
3. Services subscribed
4. Monthly and total charges
5. Churn status

Machine Learning Approach

1. Data cleaning and preprocessing
2. Exploratory data analysis (EDA)
3. Feature encoding
4. Train-test split
5. Model training using Random Forest
6. Performance evaluation
7. Model deployment
8. Model Performance

Algorithm: Random Forest Classifier

Accuracy: ~79–82% (varies slightly per run)

Evaluation metrics:

1. Accuracy
2. Precision
3. Recall
4. F1-score
4. Confusion matrix

Key Business Insights

1. Month-to-month contract customers have the highest churn.
2. Higher monthly charges correlate with increased churn risk.
3. Customers with shorter tenure are more likely to churn.
4. Long-term contracts significantly reduce churn.
5. Customers without add-on services churn more often.

Exploratory Data Analysis

Saved plots are available in:

outputs/plots/

Includes:

1. Churn distribution
2. Monthly charges vs churn
3. Tenure vs churn
4. Contract type vs churn
5. Feature importance

Technologies Used

1. Python
2. Pandas
3. NumPy
4. Scikit-learn
5. Matplotlib
6. Seaborn
7. Flask
8. Gunicorn

Future Improvements

1. Hyperparameter tuning
2. Model comparison (XGBoost, Logistic Regression)
3. Customer segmentation
4. Real-time data integration
5. Dashboard with business KPIs

Author

Nihal
B.Tech IT Student
Aspiring Data Science Intern
