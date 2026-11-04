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

---

## Task 2: Customer Segmentation

Group mall customers into segments based on Annual Income and Spending Score using the Mall Customers dataset (200 customers, 5 columns).

### What I did
- Cleaned the data and checked for missing values and duplicates (none found)
- Explored the distributions of Age, Income and Spending Score
- Scaled Income and Spending Score with StandardScaler
- Chose the number of clusters with the elbow method and silhouette score
- Trained K-Means and visualized the segments in 2D with centroids
- Analysed the average income, spending and age of each cluster

### Results
| Metric | Value |
|---|---|
| Number of clusters (K) | 5 |
| K-Means silhouette score | 0.555 |

| Cluster | Customers | Avg Income (k$) | Avg Spending | Avg Age | Segment |
|---|---|---|---|---|---|
| 0 | 81 | 55.3 | 49.5 | 42.7 | Standard customers |
| 1 | 39 | 86.5 | 82.1 | 32.7 | Target customers (high income, high spending) |
| 2 | 22 | 25.7 | 79.4 | 25.3 | Impulsive young spenders |
| 3 | 35 | 88.2 | 17.1 | 41.1 | Careful high earners |
| 4 | 23 | 26.3 | 20.9 | 45.2 | Budget customers |

### Key findings
- Both the elbow method and the silhouette score support K = 5.
- Cluster 1 (high income, high spending) is the most valuable segment for premium offers.
- Cluster 3 earns as much as Cluster 1 but spends far less, so it is the best target for loyalty programs and personalized promotions.
- Cluster 2 is young with low income but high spending, so discounts and trend-based campaigns suit it best.

### Bonus
- **DBSCAN:** With eps = 0.4 and min_samples = 5, DBSCAN found 4 clusters and labeled 15 customers as noise. Its silhouette score (0.478, noise excluded) was lower than K-Means (0.555), so K-Means fits this dataset better. DBSCAN does not need K, but it is sensitive to eps and min_samples.
- **Average spending per cluster:** Cluster 1 has the highest average spending score (82.1) and Cluster 3 the lowest (17.1), even though both have high income.

### How to run
Open `customer_segmentation.ipynb` in Google Colab, upload
`Mall_Customers.csv` to `/content/`, and run all cells.

Mall Customer Segmentation (Kaggle): https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## Task 3: Loan Approval Prediction

Predict whether a loan application will be approved using the Loan Approval Prediction dataset (4,269 applications, 13 columns).

### What I did
- Cleaned the data (stripped extra spaces, dropped the `loan_id` identifier) and checked for missing values and duplicates (none found)
- Explored the class balance, CIBIL score, income, loan amount and categorical features
- Built a preprocessing pipeline: imputation, scaling and one-hot encoding
- Used a stratified 80/20 train/test split
- Trained Logistic Regression and a Decision Tree, with and without imbalance handling
- Evaluated with precision, recall, F1-score, ROC-AUC and confusion matrices

### Results
Approved: 62.2%, Rejected: 37.8%. Scores below are for the test set (854 rows).

| Model | Accuracy | Precision (Rejected) | Recall (Rejected) | F1 (Rejected) | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Decision Tree + SMOTE | 0.971 | 0.946 | 0.978 | 0.962 | 0.969 | 0.992 |
| Decision Tree | 0.968 | 0.946 | 0.972 | 0.959 | 0.967 | 0.988 |
| Logistic Regression (balanced) | 0.924 | 0.877 | 0.929 | 0.902 | 0.920 | 0.973 |
| Logistic Regression + SMOTE | 0.923 | 0.879 | 0.923 | 0.900 | 0.919 | 0.973 |
| Logistic Regression | 0.915 | 0.903 | 0.867 | 0.885 | 0.908 | 0.973 |

### Key findings
- CIBIL score is the most important feature (importance 0.872), followed by loan term (0.091).
- The Decision Tree outperformed Logistic Regression on every metric.
- The dataset has no missing values, so the imputers in the pipeline did not change any data.
- The class imbalance is mild (62/38), so SMOTE had a small effect.

### Bonus
- **SMOTE and class weights:** For Logistic Regression they raised the recall of the Rejected class (0.867 to 0.923 and 0.929) at the cost of precision (0.903 to about 0.88). For the Decision Tree, SMOTE improved Macro F1 only slightly (0.967 to 0.969), which is too small to be conclusive on a single test split.
- **Logistic Regression vs Decision Tree:** The Decision Tree reached Macro F1 0.967 against 0.908 for Logistic Regression.

### How to run
Open `Loan_Approval_Prediction.ipynb` in Google Colab, upload
`loan_approval_dataset.csv` to `/content/`, and run all cells.

Loan Approval Prediction Dataset (Kaggle): https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

---

## Task 4: Industrial Predictive Maintenance

Predict machine failures from sensor readings on the AI4I 2020 Predictive Maintenance dataset (10,000 records, 3.39% failures), with a focus on a low False Discovery Rate (FDR = false alarms / all alarms).

### What I did
- Cleaned the data and built a `failure_type` label from the failure-mode flags
- Explored sensor distributions, failure rate by product type and correlations
- Engineered three features: temperature difference, mechanical power and tool wear x torque
- Trained Logistic Regression, Random Forest and XGBoost with class weights (cost-sensitive learning)
- Chose the decision threshold with 5-fold cross-validation on the training set (lowest FDR while keeping recall at or above 70%)
- Evaluated once on the test set (68 failures) and compared with an Isolation Forest (anomaly detection)
- Trained a multi-class model to predict the failure type

### Results
| Model | Setting | Precision | Recall | FDR | False alarms | Missed failures |
|---|---|---|---|---|---|---|
| Random Forest | Default (0.5) | 0.947 | 0.794 | 0.053 | 3 | 14 |
| Random Forest | Tuned (0.613) | 0.961 | 0.721 | 0.039 | 2 | 19 |
| XGBoost | Default (0.5) | 0.718 | 0.824 | 0.282 | 22 | 12 |
| XGBoost | Tuned (0.969) | 0.939 | 0.676 | 0.061 | 3 | 22 |
| Logistic Regression | Default (0.5) | 0.179 | 0.882 | 0.821 | 275 | 8 |
| Logistic Regression | Tuned (0.703) | 0.292 | 0.794 | 0.708 | 131 | 14 |

Isolation Forest (unsupervised): precision 0.123, recall 0.103, FDR 0.877.

### Failure type classification
| Failure type | F1 | FDR | Test samples |
|---|---|---|---|
| Heat Dissipation | 0.977 | 0.045 | 21 |
| Overstrain | 0.970 | 0.059 | 16 |
| Power | 0.968 | 0.000 | 16 |
| Multiple failures | 0.800 | 0.200 | 5 |
| Tool Wear | 0.000 | 1.000 | 8 |

### Key findings
- Random Forest had the lowest cross-validated FDR (0.020) and was selected as the best model.
- Threshold tuning cut XGBoost's false alarms from 22 to 3. For Random Forest it removed only one false alarm and cost five missed failures, so the best threshold depends on the real cost of each error.
- Logistic Regression and Isolation Forest produced too many false alarms to be useful.
- Tool Wear failures were not detected by the failure type model (0 of 8), so overall accuracy (0.994) hides a macro F1 of 0.673.
- Torque is the sensor most associated with failure (AUC 0.770), followed by rotational speed (0.767).

### Limitations
- The dataset is synthetic, and the test set has only 68 failures, so one false alarm changes the FDR by about 2 percentage points.
- The costs used in the cost analysis (5,000 per false alarm, 20,000 per missed failure) are assumptions.
- The sensor ranking shows association, not which sensor changes first before a failure, because there are no timestamps.

### Bonus
- **Sensor correlation analysis:** done (torque is the strongest single indicator).
- **Time-to-Failure regression:** not done, because the dataset has no time-to-failure or remaining-useful-life column.

### How to run
Open `predictive_maintenance.ipynb` in Google Colab, upload the dataset CSV to `/content/` (the notebook reads `predictive_maintenance_datasets.csv`, which is `ai4i2020.csv` renamed), and run all cells.

AI4I 2020 Predictive Maintenance Dataset (Kaggle): https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
