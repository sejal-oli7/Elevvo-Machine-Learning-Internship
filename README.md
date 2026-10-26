
# Elevvo Machine Learning Internship

## Task 1: Student Exam Score Prediction

Predict a student's exam score from study habits and background factors
using the Student Performance Factors dataset (6,607 students, 20 columns).

### What I did
- Explored the data (correlations, outliers, missing values)
- Built a preprocessing pipeline: imputation, ordinal and one-hot encoding, scaling
- Trained Linear Regression and compared it with Ridge, Huber and Random Forest
- Evaluated with MAE, RMSE, R2 and 5-fold cross-validation
- Analysed the prediction errors

### Results
| Metric | Value |
|---|---|
| Test R2 (all rows) | 0.7709 |
| Test MAE | 0.44 |
| Test RMSE | 1.80 |
| 5-fold CV R2 | 0.7376 (+/- 0.10) |
| Test R2 (rows with score < 80) | 0.9862 |

Attendance and Hours_Studied are the most important features.
Seven high-score students (80-98) in the test set were predicted poorly
and account for most of the error. Their features look ordinary, so these
scores are not explained by the available data.

### How to run
Open `student_score_prediction.ipynb` in Google Colab, upload
`StudentPerformanceFactors.csv` to `/content/`, and run all cells.

### Dataset
Student Performance Factors (Kaggle).  Link : https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
