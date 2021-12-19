# FinalProjectMachineLearning
Kaggle Competiiton
https://www.kaggle.com/c/ilp2021f


# Task description
This dataset was extracted by by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the specific conditions. Prediction task is to determine whether a person makes over 50K a year given the census information. There are 14 attributes, including continuous, categorical and integer types. Some attributes may have missing values, recorded as question marks.

# Evaluation
The evaluation is based on Area Under ROC (AUC) curve, which is a value between 0 and 1. The higher AUC, the better the predictive performance. Note that AUC is the most commonly used measure in ML practice. It considers the cases of all possible thresholds that are used for (binary) classification, and calculates the area of the (TPR, FPR) curve of using these thresholds (TPR and FPR stands for True Positive Rate and False Positive Rate, respectively) as an overall measure of the model performance. Therefore, AUC is not restricted to the accuracy of any single threshold (e.g., 0.5 or 0). It is a comprehensive evaluation. A detailed introduction of AUC is given here.

The file should contain a header and have the following format:

ID,Prediction
3,0.3
8,-0.2
10,0.55
...


# IncomeLvlPred-Midterm.py
Midterm Implementations

# IncomeLvlPred-Final.py
Final Implementations
