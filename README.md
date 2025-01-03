# Stroke Prediction and Analysis
## Overview
This project aims to predict stroke occurrence using health-related features. It involves data preprocessing, exploratory data analysis, and machine learning modeling using a Random Forest algorithm.
### Dataset
The dataset "healthcare-dataset-stroke-data.csv" contains various health-related features including age, gender, hypertension status, heart disease status, BMI, glucose levels, and smoking status, and was sourced from a course project network.
### Data Processing Steps
The dataset was loaded and split into training and testing sets( 80:20 respectively), followed by handling missing values in the BMI column through median imputation;converting categorical variables into factors, applying log transformations to the BMI and average glucose levels, and performing min-max scaling on relevant features.
### Model Training
A Random Forest model was trained using the training dataset with weights applied to address class imbalance, ensuring that the minority class (stroke cases) is given more importance during model fitting
### Results
The model's performance is evaluated using a confusion matrix, which provides metrics such as accuracy, precision, and recall, along with a feature importance plot that highlights the key predictors of stroke risk; additionally, the ROC curve analysis indicates good discriminative ability with an AUC greater than 0.80
