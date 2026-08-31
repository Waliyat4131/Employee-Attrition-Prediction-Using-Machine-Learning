# Employee-Attrition-Prediction-Using-Machine-Learning
Predicting Employee Attrition using machine learning


Project Overview

Employee attrition can create recruitment, training, productivity, and knowledge-retention challenges for organisations. This project develops and evaluates machine learning classification models to predict whether an employee is likely to leave an organisation using demographic, job-related, compensation, and workplace characteristics.

The project uses the IBM HR Analytics Employee Attrition and Performance dataset and compares six machine learning approaches, including a neural network.

Business Problem

Can employee demographic, job, compensation, and workplace characteristics be used to predict employee attrition?

The aim is to identify patterns associated with employee turnover and develop a predictive model that could support proactive HR retention decisions.

Important: This project demonstrates predictive modelling and should not be used as an automated decision-making system for employment actions. Predictions should be treated as analytical support and interpreted with appropriate human oversight.

Dataset

The project uses the IBM HR Analytics Employee Attrition and Performance dataset sourced from Kaggle.

Observations: 1,470 employees

Original variables: 35

Target: Attrition

Classes: No (stayed) and Yes (left)

Approximate target distribution: 84% No / 16% Yes

The class imbalance was considered during model evaluation, so accuracy was assessed alongside precision, recall, F1-score, and ROC-AUC.

Project Workflow

Raw Dataset
     ↓
Data Cleaning & Quality Checks
     ↓
Exploratory Data Analysis
     ↓
Train / Validation / Test Strategy
     ↓
Categorical Encoding & Feature Scaling
     ↓
Model Development
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Final Model Selection

Data Preparation

The analysis included:

Missing-value checks

Duplicate checks

Data-type checks

Removal of constant/unnecessary columns

Outlier analysis

One-hot encoding of categorical variables

Standardisation of numerical features

Train/validation/test splitting

Examination of class imbalance

The analysis found no null values or duplicate records. Several realistic high-value observations were identified as outliers, particularly in income and employment-history variables; these were retained because they represent plausible employee characteristics rather than confirmed data-entry errors.

Exploratory Data Analysis

The analysis identified several notable patterns:

Younger employees appeared more likely to leave than older employees.

Employees with lower monthly income showed higher observed attrition.

Employees with shorter organisational tenure appeared more likely to leave.

Employees working overtime showed higher observed attrition.

Research & Development, Sales, and Human Resources accounted for substantial numbers of observed departures.

Monthly income and job level showed a strong positive relationship.

These are associations in the dataset, not causal conclusions.

Machine Learning Models

Six classification approaches were evaluated:

Logistic Regression

Decision Tree

Random Forest

Naïve Bayes

Gradient Boosting

Neural Network

The models were selected to provide a comparison between interpretable linear modelling, tree-based methods, probabilistic classification, ensemble learning, and a neural-network approach.

Model Evaluation

The models were assessed using:

Accuracy — overall proportion of correct predictions

Precision — correctness of positive predictions

Recall — ability to identify relevant positive cases

F1-score — balance between precision and recall

ROC-AUC — ability to discriminate between the two classes across classification thresholds

Performance Summary

Model

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Logistic Regression

87.33%

85.99%

87.33%

85.51%

0.815

Neural Network

87.33%

86.13%

87.33%

85.22%

0.780

Gradient Boosting

85.07%

82.72%

85.07%

81.10%

0.813

Random Forest

84.62%

81.60%

84.62%

81.24%

0.827

Decision Tree

79.19%

76.70%

79.19%

77.78%

0.562

Naïve Bayes

66.06%

82.55%

66.06%

70.47%

0.753

Key Results

Logistic Regression

Logistic Regression achieved the highest overall accuracy jointly with the Neural Network at 87.33% and achieved the highest F1-score at 85.51%. Its ROC-AUC of 0.815 also indicates strong class discrimination.

Random Forest

Random Forest achieved the highest ROC-AUC at 0.827, indicating the strongest discrimination between employees who left and those who stayed among the evaluated models. However, its accuracy and F1-score were lower than Logistic Regression.

Neural Network

The Neural Network achieved 87.33% accuracy and an F1-score of 85.22%. Its ROC-AUC was 0.780, lower than Logistic Regression and Random Forest.

Gradient Boosting

Gradient Boosting achieved 85.07% accuracy, an F1-score of 81.10%, and an ROC-AUC of 0.813, showing competitive predictive and discriminatory performance.

Decision Tree

Decision Tree achieved 79.19% accuracy and an ROC-AUC of 0.562, making it the weakest model in terms of class discrimination among the evaluated models.

Naïve Bayes

Naïve Bayes achieved 66.06% accuracy, although its precision was relatively high at 82.55%. Its overall performance was weaker than the other models based on the combined evaluation metrics.

Hyperparameter Tuning

Hyperparameter tuning was performed using RandomizedSearchCV.

For Logistic Regression, the search included:

C: 0.1, 1, 10, 100

solver: lbfgs, liblinear

The tuning process used cross-validation and F1-score as the optimisation metric. This is important when interpreting the results: tuning for F1-score does not guarantee an improvement in ROC-AUC because the two metrics evaluate different aspects of model performance.

The selected Logistic Regression configuration was:

C = 0.1
solver = liblinear

Feature Importance

For the Random Forest model, the five highest feature-importance values identified in the analysis were:

Rank

Feature

Importance

1

MonthlyIncome

0.100247

2

Age

0.065476

3

TotalWorkingYears

0.062920

4

OverTime_Yes

0.060613

5

YearsAtCompany

0.049176

These values indicate which variables contributed most to the Random Forest's predictive process. Feature importance should not be interpreted as proof that a variable causes attrition.

Final Model Selection

Logistic Regression was selected as the preferred model because it provided the strongest overall balance across the main classification metrics while remaining relatively simple and interpretable.

Random Forest produced the highest ROC-AUC (0.827), but Logistic Regression achieved higher accuracy and F1-score in the reported evaluation. Therefore, the final choice reflects a balance between predictive performance and interpretability rather than selecting a model solely on the highest single metric.

Business Insights

The analysis suggests several areas that could be investigated by HR teams:

Review workload and overtime patterns.

Examine compensation and career-development opportunities.

Pay particular attention to early-tenure employees.

Investigate retention patterns across job roles and departments.

Use predictive analytics alongside, rather than instead of, employee engagement and HR expertise.

Limitations

The dataset contains 1,470 observations and may not represent all organisations or industries.

The data is a benchmark HR dataset rather than evidence from a live organisational workforce.

Observed relationships do not establish causation.

Class imbalance may influence some evaluation metrics.

Model predictions should not be interpreted as certain outcomes.

Feature importance does not establish causal influence.

Further external validation would be required before applying a model to a real organisation.

Future Improvements

Potential extensions include:

Testing additional class-imbalance strategies.

Optimising models against alternative metrics such as ROC-AUC or recall.

Applying explainability techniques such as SHAP.

Calibrating predicted probabilities.

Testing the models on additional HR datasets.

Building an interactive prediction application/API.

Monitoring model performance and fairness across relevant employee groups before any real-world use.

Technologies

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

TensorFlow / Keras

Jupyter Notebook

Git / GitHub

Repository Structure

employee-attrition-prediction/
│
├── README.md
├── notebooks/
│   └── employee_attrition_prediction.ipynb
├── images/
│   ├── eda.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── roc_curves.png
├── requirements.txt
└── .gitignore

Author

Waliyat Morenike

Data Science | Artificial Intelligence & Machine Learning | HR Analytics

Dataset Source

IBM HR Analytics Employee Attrition & Performance dataset, available through Kaggle.
