import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

def load_data(landmarks_path, labels_path):
    """Load the dataset."""
    X = np.load(landmarks_path)
    y = np.load(labels_path)
    return X, y

def strategy_a_minmax(X):
    """
    Strategy A: Min-Max Scaling (Baseline).
    The provided dataset 'your_landmarks.npy' is already normalized via Min-Max.
    We simply return it as is.
    """
    return X.copy()

def strategy_b_zscore(X):
    """
    Strategy B: Center of Mass (Z-Score) Normalization.
    For each sample of 21x3 coordinates, we compute the mean and std,
    then standardize so the 'center of mass' is at (0,0,0) and variance is 1.
    """
    X_new = []
    for row in X:
        coords = row.reshape(21, 3)
        mean = np.mean(coords, axis=0) # shape (3,)
        std = np.std(coords, axis=0)   # shape (3,)
        std[std == 0] = 1e-6           # prevent division by zero
        normalized = (coords - mean) / std
        X_new.append(normalized.flatten())
    return np.array(X_new)

def strategy_c_anatomical(X):
    """
    Strategy C: Anatomical Distance Scaling.
    The data has the wrist at (0,0,0). We find the distance to the middle finger knuckle (index 9)
    and divide all coordinates by this distance.
    In the flattened 63-element array, index 9 corresponds to elements 27, 28, 29.
    """
    X_new = []
    for row in X:
        # middle_mcp is at indices 27, 28, 29
        middle_mcp = row[27:30]
        # Calculate Euclidean distance from origin (wrist)
        dist = np.linalg.norm(middle_mcp)
        if dist < 1e-6:
            dist = 1e-6 # prevent division by zero
        X_new.append(row / dist)
    return np.array(X_new)

def evaluate_model(X, y, strategy_name):
    """Train a Random Forest model and return metrics + predictions."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y_train)))
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }
    
    return metrics, cm, labels

def plot_bar_chart(results, save_path):
    """Generate a grouped bar chart for all metrics across strategies."""
    df_data = []
    for strategy, metrics in results.items():
        for metric_name, value in metrics.items():
            df_data.append({"Strategy": strategy, "Metric": metric_name, "Score": value})
            
    df = pd.DataFrame(df_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Metric", y="Score", hue="Strategy", palette="viridis")
    plt.title("Performance Comparison of Normalization Strategies")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, labels, strategy_name, save_path):
    """Generate and save a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {strategy_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("🚀 Starting Empirical Evaluation of Normalization Strategies...")
    
    landmarks_path = os.path.join("..", "your_landmarks.npy")
    labels_path = os.path.join("..", "your_labels.npy")
    
    if not os.path.exists(landmarks_path) or not os.path.exists(labels_path):
        print(f"❌ Error: Could not find datasets at {landmarks_path} and {labels_path}")
        sys.exit(1)
        
    X_raw, y = load_data(landmarks_path, labels_path)
    print(f"✅ Loaded dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
    
    strategies = {
        "A: Min-Max (Baseline)": strategy_a_minmax,
        "B: Z-Score (Center of Mass)": strategy_b_zscore,
        "C: Anatomical Distance": strategy_c_anatomical
    }
    
    results = {}
    
    for name, func in strategies.items():
        print(f"\nProcessing {name}...")
        X_processed = func(X_raw)
        
        metrics, cm, labels = evaluate_model(X_processed, y, name)
        results[name] = metrics
        
        # Save Confusion Matrix
        safe_name = name.split(":")[0].replace(" ", "_")
        cm_path = f"confusion_matrix_{safe_name}.png"
        plot_confusion_matrix(cm, labels, name, cm_path)
        print(f"   Accuracy: {metrics['Accuracy']:.4f}  |  F1: {metrics['F1 Score']:.4f}")
        
    # Save Bar Chart
    plot_bar_chart(results, "accuracy_comparison.png")
    
    # Save metrics to JSON
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n✅ Experiments completed successfully! Visualizations saved in 'research/' directory.")

if __name__ == "__main__":
    main()
