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
