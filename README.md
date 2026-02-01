# Credit Card Dataset Sampling and Model Performance Analysis

**Author:** Roll No. 102303059  
**Date:** February 1, 2026

## 1. Introduction

This project explores how different sampling techniques can help balance a highly imbalanced credit card fraud dataset and how these techniques affect the performance of various machine learning models. The analysis compares **5 sampling techniques** across **7 classification models** to determine the best combination for handling class imbalance.

## 2. Dataset Description

The dataset `Creditcard_data.csv` contains credit card transaction records with a binary class label:

- **Total Samples:** 772
- **Class 0 (Legitimate transactions):** 763 samples (98.83%)
- **Class 1 (Fraudulent transactions):** 9 samples (1.17%)
- **Imbalance Ratio:** 84.78:1

This severe imbalance motivates the use of sampling techniques before model training.

## 3. Sampling Process

- **Target Population:** All credit card transactions in the dataset
- **Sampling Frame:** The complete dataset file (`Creditcard_data.csv`)
- **Sample Size:** 9 samples from each class (balanced dataset of 18 samples total)
- **Random State:** 42 (for reproducibility)

## 4. Sampling Techniques Used

Five sampling techniques were applied to create balanced datasets:

1. **Sampling1 - Simple Random Sampling**  
   Randomly selects equal numbers of samples from each class without any systematic approach.

2. **Sampling2 - Systematic Sampling**  
   Selects samples at fixed intervals after shuffling, providing better coverage than pure random sampling.

3. **Sampling3 - Stratified Sampling**  
   Ensures equal representation from each class while preserving the overall structure of the dataset.

4. **Sampling4 - Cluster Sampling**  
   Divides data into clusters, shuffles them, and selects samples to form a balanced dataset.

5. **Sampling5 - Bootstrap Sampling**  
   Uses sampling with replacement to generate balanced class distributions, allowing diversity in training sets.

Each technique results in a perfectly balanced dataset of 18 samples (9 per class).

## 5. Machine Learning Models

Seven classification models were evaluated:

- **M1:** Logistic Regression (max_iter=1000, n_jobs=-1)
- **M2:** Decision Tree (default parameters)
- **M3:** Random Forest (n_estimators=100, n_jobs=-1)
- **M4:** Support Vector Machine (SVC with default parameters)
- **M5:** K-Nearest Neighbors (n_neighbors=5, n_jobs=-1)
- **M6:** XGBoost (eval_metric='logloss', n_jobs=-1)
- **M7:** CatBoost (verbose=0, thread_count=-1)

## 6. Experimental Setup

- **Train-test split:** 70% training, 30% testing
- **Stratified split:** Maintains class balance in train/test sets
- **Random seed:** 42 (for reproducibility)
- **Evaluation metric:** Accuracy
- **Total experiments:** 35 (5 sampling techniques × 7 models)

## 7. Results

### Accuracy Table

| Model | Sampling1_SimpleRandom | Sampling2_Systematic | Sampling3_Stratified | Sampling4_Cluster | Sampling5_Bootstrap |
|-------|------------------------|----------------------|----------------------|-------------------|---------------------|
| **M1_LogisticRegression** | 0.6667 | 0.3333 | 0.6667 | 0.6667 | 0.6667 |
| **M2_DecisionTree** | 0.5000 | 0.6667 | 0.5000 | 0.5000 | **0.8333** |
| **M3_RandomForest** | 0.5000 | 0.3333 | 0.5000 | 0.6667 | 0.6667 |
| **M4_SVM** | 0.5000 | 0.1667 | 0.5000 | 0.3333 | 0.6667 |
| **M5_KNN** | 0.5000 | 0.3333 | 0.5000 | 0.3333 | 0.6667 |
| **M6_XGBoost** | 0.3333 | 0.1667 | 0.3333 | 0.5000 | 0.6667 |
| **M7_CatBoost** | 0.5000 | 0.5000 | 0.5000 | 0.5000 | **0.8333** |

## 8. Key Observations

1. **Bootstrap Sampling (Sampling5) performs best overall:** It achieved the highest accuracy across most models, particularly excelling with Decision Tree and CatBoost (0.8333).

2. **Model Performance Varies by Sampling Technique:**
   - Logistic Regression performs consistently well except with Systematic Sampling
   - Tree-based models (Decision Tree, Random Forest, CatBoost) benefit most from Bootstrap Sampling
   - SVM and KNN show significant improvement with Bootstrap Sampling

3. **Systematic Sampling shows mixed results:** While it worked well for Decision Tree, it performed poorly for most other models.

4. **Gradient Boosting Models:** Both XGBoost and CatBoost achieved their best performance with Bootstrap Sampling, with CatBoost reaching the highest accuracy.

5. **Small Sample Size Impact:** Due to the extremely small balanced dataset (18 samples with 6 test samples), accuracy changes in discrete steps (multiples of ~16.67%).

## 9. Best Performing Combinations

### Best Sampling Technique per Model

| Model | Best Sampling | Accuracy |
|-------|--------------|----------|
| M1_LogisticRegression | Sampling1_SimpleRandom | 0.6667 |
| M2_DecisionTree | Sampling5_Bootstrap | **0.8333** |
| M3_RandomForest | Sampling4_Cluster | 0.6667 |
| M4_SVM | Sampling5_Bootstrap | 0.6667 |
| M5_KNN | Sampling5_Bootstrap | 0.6667 |
| M6_XGBoost | Sampling5_Bootstrap | 0.6667 |
| M7_CatBoost | Sampling5_Bootstrap | **0.8333** |

### Best Overall Combination

- **Model:** Decision Tree (M2) or CatBoost (M7)
- **Sampling Technique:** Bootstrap Sampling (Sampling5)
- **Accuracy:** 0.8333

## 10. Conclusions

1. **Bootstrap Sampling is the most robust technique** for this imbalanced dataset, performing well across 6 out of 7 models.

2. **Tree-based models** (Decision Tree and CatBoost) achieved the highest accuracy when combined with Bootstrap Sampling.

3. **Sampling technique selection matters:** The choice of sampling technique can result in accuracy differences of up to 50% for the same model.

4. **Modern gradient boosting algorithms** (XGBoost, CatBoost) show promise but require appropriate sampling techniques to reach their potential.

## 11. Requirements

```
pandas
numpy
scikit-learn
imblearn
xgboost
catboost
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 12. How to Run

```bash
python sampling_analysis.py
```

The script will:
1. Load the credit card dataset
2. Apply all 5 sampling techniques
3. Train and evaluate all 7 models on each sampled dataset
4. Display comprehensive results including the best combinations

## 13. Project Structure

```
D:\Sampling\
├── Creditcard_data.csv          # Input dataset
├── sampling_analysis.py         # Main analysis script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
