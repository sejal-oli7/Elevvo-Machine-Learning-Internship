# Elevvo Machine Learning Internship

## Task 1: Student Exam Score Prediction

Predict a student's exam score from study habits and background factors using the Student Performance Factors dataset (6,607 students, 20 columns).

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

### Key findings
- Attendance and Hours_Studied are the most important features.
- Seven high-score students (80-98) in the test set were predicted poorly and account for most of the error. Their features look ordinary, so these scores are not explained by the available data.

### Bonus
- **Feature combinations:** Hours_Studied alone gives R2 0.23. With Attendance it rises to 0.59, and with all features to 0.77 (MAE 2.45 to 0.44). Sleep_Hours, Gender and School_Type have no effect on the score.
- **Polynomial regression:** Degrees 1-4 gave no improvement (R2 stays 0.23 on Hours_Studied only). The relationship is linear, so Linear Regression is used.

  ### How to run
Open `student_score_prediction.ipynb` in Google Colab, upload
`StudentPerformanceFactors.csv` to `/content/`, and run all cells.


Student Performance Factors (Kaggle): https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
