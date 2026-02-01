"""
Sampling Techniques Analysis for Imbalanced Credit Card Dataset

Author: Roll No. 102303059
Date: February 1, 2026

This script implements and compares five different sampling techniques
on an imbalanced credit card dataset, evaluating their effectiveness
with multiple machine learning models.

Sampling Techniques:
1. Simple Random Sampling
2. Systematic Sampling
3. Stratified Sampling
4. Cluster Sampling
5. Bootstrap Sampling

Models Used:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- XGBoost
- CatBoost
"""

import pandas as pd
import numpy as np
import random
import warnings
from typing import Dict, Tuple
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# Configuration Constants
RANDOM_STATE = 42
TEST_SIZE = 0.3
DATA_PATH = "Creditcard_data.csv"
STUDENT_ROLL_NO = "102303059"

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def load_data(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load the dataset and identify the target column.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (dataframe, target_column_name)
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the data is empty
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Identify the target column
    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    
    print("=" * 80)
    print(f"SAMPLING ANALYSIS PROJECT - Roll No: {STUDENT_ROLL_NO}")
    print("=" * 80)
    print(f"\nDataset loaded successfully: {len(df)} samples")
    print(f"Target column: {target_col}")
    print(f"\nClass distribution before sampling:")
    print(df[target_col].value_counts())
    print(f"\nImbalance ratio: {df[target_col].value_counts().max() / df[target_col].value_counts().min():.2f}:1")
    
    return df, target_col


def print_class_counts(data: pd.DataFrame, label: str, target_col: str) -> None:
    """
    Print class distribution after sampling.
    
    Args:
        data: The sampled dataframe
        label: Name of the sampling technique
        target_col: Name of the target column
    """
    print(f"\nClass distribution after {label}:")
    print(data[target_col].value_counts())
    print(f"Total samples: {len(data)}")

def simple_random_sampling(df: pd.DataFrame, target_col: str, min_class_size: int) -> pd.DataFrame:
    """
    Simple Random Sampling: Randomly select equal samples from each class.
    
    This technique ensures balanced classes by randomly selecting the same
    number of samples from each class without any systematic approach.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        min_class_size: Size of the minority class
        
    Returns:
        Balanced dataframe
    """
    samples = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        samples.append(cls_df.sample(min_class_size, random_state=RANDOM_STATE))
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def systematic_sampling(df: pd.DataFrame, target_col: str, min_class_size: int) -> pd.DataFrame:
    """
    Systematic Sampling: Select samples at fixed intervals after shuffling.
    
    This technique provides better coverage by selecting samples at regular
    intervals, which can be more representative than pure random sampling.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        min_class_size: Size of the minority class
        
    Returns:
        Balanced dataframe
    """
    samples = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        k = max(1, len(cls_df) // min_class_size)
        start = np.random.randint(0, k) if k > 1 else 0
        idx = np.arange(start, start + k * min_class_size, k)
        samples.append(cls_df.iloc[idx[:min_class_size]])
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def stratified_sampling(df: pd.DataFrame, target_col: str, min_class_size: int) -> pd.DataFrame:
    """
    Stratified Sampling: Sample proportionally from each class.
    
    Ensures representation from each class while maintaining the overall
    structure of the dataset.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        min_class_size: Size of the minority class
        
    Returns:
        Balanced dataframe
    """
    samples = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        samples.append(cls_df.sample(min_class_size, random_state=RANDOM_STATE))
    return pd.concat(samples, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def cluster_sampling(df: pd.DataFrame, target_col: str, min_class_size: int, n_clusters: int = 5) -> pd.DataFrame:
    """
    Cluster Sampling: Split each class into clusters, then select clusters.
    
    Groups data into clusters and selects entire clusters, which can be
    computationally efficient for large datasets.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        min_class_size: Size of the minority class
        n_clusters: Number of clusters to create
        
    Returns:
        Balanced dataframe
    """
    samples = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        # Split into clusters and shuffle
        cluster_size = len(cls_df) // n_clusters
        clusters = [cls_df.iloc[i*cluster_size:(i+1)*cluster_size] for i in range(n_clusters)]
        if len(cls_df) % n_clusters != 0:
            clusters.append(cls_df.iloc[n_clusters*cluster_size:])
        random.shuffle(clusters)
        # Concatenate clusters and select required samples
        selected = pd.concat(clusters, ignore_index=True)
        samples.append(selected.iloc[:min_class_size])
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def bootstrap_sampling(df: pd.DataFrame, target_col: str, min_class_size: int) -> pd.DataFrame:
    """
    Bootstrap Sampling: Sample with replacement to balance classes.
    
    Allows the same instance to be selected multiple times, which can be
    useful for creating diverse training sets and estimating variance.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        min_class_size: Size of the minority class
        
    Returns:
        Balanced dataframe
    """
    samples = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        samples.append(
            resample(
                cls_df,
                replace=True,
                n_samples=min_class_size,
                random_state=RANDOM_STATE
            )
        )
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def create_balanced_datasets(df: pd.DataFrame, target_col: str, min_class_size: int) -> Dict[str, pd.DataFrame]:
    """
    Create balanced datasets using various sampling techniques.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        min_class_size: Size of the minority class
        
    Returns:
        Dictionary of sampling technique name to balanced dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING BALANCED DATASETS")
    print("=" * 80)
    
    datasets = {
        "Sampling1_SimpleRandom": simple_random_sampling(df, target_col, min_class_size),
        "Sampling2_Systematic": systematic_sampling(df, target_col, min_class_size),
        "Sampling3_Stratified": stratified_sampling(df, target_col, min_class_size),
        "Sampling4_Cluster": cluster_sampling(df, target_col, min_class_size),
        "Sampling5_Bootstrap": bootstrap_sampling(df, target_col, min_class_size)
    }
    
    for name, data in datasets.items():
        print_class_counts(data, name, target_col)
    
    return datasets


def get_models() -> Dict[str, object]:
    """
    Define and return machine learning models for evaluation.
    
    Returns:
        Dictionary of model name to model instance
    """
    return {
        "M1_LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        "M2_DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "M3_RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "M4_SVM": SVC(random_state=RANDOM_STATE),
        "M5_KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "M6_XGBoost": XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'),
        "M7_CatBoost": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, thread_count=-1)
    }


def evaluate_models(datasets: Dict[str, pd.DataFrame], 
                    models: Dict[str, object], 
                    target_col: str) -> pd.DataFrame:
    """
    Train and evaluate each model on each sampled dataset.
    
    Args:
        datasets: Dictionary of sampling technique to dataframe
        models: Dictionary of model name to model instance
        target_col: Target column name
        
    Returns:
        DataFrame containing accuracy results for all model-sampling combinations
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    results = pd.DataFrame(index=models.keys(), columns=datasets.keys())
    
    for sample_name, sample_df in datasets.items():
        print(f"\nEvaluating models on {sample_name}...")
        
        X = sample_df.drop(columns=[target_col])
        y = sample_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                results.loc[model_name, sample_name] = round(acc, 4)
                print(f"  {model_name}: {acc:.4f}")
            except Exception as e:
                print(f"  {model_name}: Error - {str(e)}")
                results.loc[model_name, sample_name] = 0.0
    
    return results


def display_results(results: pd.DataFrame) -> None:
    """
    Display comprehensive results and identify best combinations.
    
    Args:
        results: DataFrame containing accuracy results
    """
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nStudent Roll No: {STUDENT_ROLL_NO}")
    print(f"\nAccuracy Results Table:")
    print(results.to_string())
    
    # Best sampling for each model
    print("\n" + "-" * 80)
    print("Best Sampling Technique for Each Model:")
    print("-" * 80)
    for model in results.index:
        best_sampling = results.loc[model].astype(float).idxmax()
        best_acc = results.loc[model].astype(float).max()
        print(f"{model:30s} → {best_sampling:25s} (Accuracy: {best_acc:.4f})")
    
    # Overall best combination
    best_model = results.max(axis=1).astype(float).idxmax()
    best_model_acc = results.max(axis=1).astype(float).max()
    best_sampling_for_best_model = results.loc[best_model].astype(float).idxmax()
    
    print("\n" + "=" * 80)
    print("BEST OVERALL COMBINATION")
    print("=" * 80)
    print(f"Model: {best_model}")
    print(f"Sampling Technique: {best_sampling_for_best_model}")
    print(f"Accuracy: {best_model_acc:.4f}")
    print("=" * 80)


def main():
    """
    Main execution function for sampling analysis.
    """
    try:
        # Load data
        df, target_col = load_data(DATA_PATH)
        
        # Calculate minimum class size
        min_class_size = df[target_col].value_counts().min()
        
        # Create balanced datasets
        datasets = create_balanced_datasets(df, target_col, min_class_size)
        
        # Get models
        models = get_models()
        
        # Evaluate models
        results = evaluate_models(datasets, models, target_col)
        
        # Display results
        display_results(results)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'Creditcard_data.csv' exists in the current directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
