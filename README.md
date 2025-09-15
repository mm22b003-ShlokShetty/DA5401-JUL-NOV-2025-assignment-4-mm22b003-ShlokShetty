# DA5401 Assignment 4 - GMM-Based Synthetic Sampling

Notebook(s) for Assignment 4. Dataset (creditcard.csv) is NOT included. Download from Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud

Run the notebook inside a Python virtual environment.

# DA5401 – Assignment A4: GMM-Based Balancing for Imbalanced Data

## Overview
This assignment addresses the problem of **class imbalance** in a credit card fraud dataset (or similarly imbalanced dataset).  
The majority class dominates the dataset, leading standard classifiers to ignore the minority class.  
To tackle this, we explored **Gaussian Mixture Model (GMM)-based synthetic oversampling** and **Clustering-Based Undersampling (CBU)** techniques to balance the training data.

---

## Problem Setup
- **Dataset:** Highly imbalanced (majority ≫ minority).  
- **Split:** Train/test split with scaling applied (Part A).  
- **Task:** Train Logistic Regression classifiers on rebalanced training sets and evaluate performance on the **original imbalanced test set**.

---

## Methods

### Part B.2 – GMM Model Selection
- Fit GMMs on the **minority class** for `k = 1..8`.  
- Chose the best number of components (`best_k`) using **BIC**.  
- This `final_gmm` was used for synthetic data generation.

### Part B.3 – Synthetic Oversampling with GMM
- Generated synthetic minority samples from the fitted GMM.  
- Oversampled until the **minority matched the majority** size. 

### Part B.4 – Clustering-Based Undersampling (CBU) + GMM Oversampling
- Applied **MiniBatchKMeans** clustering on the majority class (`k_major ≈ 50` chosen via elbow method).  
- Reduced majority to ~20% of original size (~45,490 samples).  
- Used GMM to oversample minority to the same size.  


---

## Results

Both models were trained on the rebalanced sets and evaluated on the **original imbalanced test set** (56,864 majority, 98 minority).  

| Model                  | Minority Precision | Minority Recall | Minority F1 | ROC AUC | Notes |
|-------------------------|-------------------|-----------------|-------------|---------|-------|
| **Baseline (Imbalanced)** | ~0.00            | ~0.00           | ~0.00       | 0.50    | Model ignores minority |
| **GMM-only (B.3)**     | 0.079             | **0.898**       | 0.146       | **0.967** | Huge recall gain, many false positives |
| **CBU + GMM (B.4)**    | **0.105**         | 0.888           | **0.188**   | 0.963   | Better precision–recall trade-off, faster training |

---

## Analysis
- **Baseline:** Logistic Regression on imbalanced data fails to detect the minority class.  
- **GMM-only oversampling:** Excellent recall (~90%), but very low precision. The model detects almost all frauds but with many false alarms.  
- **CBU + GMM:** Slightly lower recall but improved precision and F1. Also reduced training size (90k vs. 450k), making the approach more efficient.  

**Key insight:**  
Oversampling with GMM improves the model’s ability to learn the minority distribution. However, balancing both **by undersampling the majority (CBU)** and **oversampling the minority (GMM)** yields a better overall classifier with fewer false positives.

---

## Recommendation
We recommend using **CBU + GMM (Part B.4)** for this dataset.  
- It achieves a **better F1-score** for the minority class.  
- Provides a balanced trade-off between precision and recall.  
- Is computationally more efficient (smaller training set).  

**Conclusion:**  
> GMM-based synthetic oversampling is effective for minority detection, but combining it with CBU produces a more robust and practical solution.

---

## 📈 Visualizations
- **Elbow plot** for GMM component selection (Part B.2).  
- **Elbow plot** for `k_major` in majority clustering.  
- **Confusion matrix heatmaps** for both B.3 and B.4 results.  


---
