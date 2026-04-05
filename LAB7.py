
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# These are optional - install them with: pip install xgboost catboost shap lime
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("xgboost not installed, skipping it")

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False
    print("catboost not installed, skipping it")

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("shap not installed, skipping SHAP section")

try:
    import lime
    import lime.lime_tabular
    LIME_OK = True
except ImportError:
    LIME_OK = False
    print("lime not installed, skipping LIME section")


# -----------------------------------------------------------
# Load your dataset here
# -----------------------------------------------------------
data = load_iris(as_frame=True)
X = data.data
y = data.target
target_names = data.target_names
feature_names = list(X.columns)

print(f"Shape: {X.shape}")
print(f"Classes: {list(target_names)}")
print(f"Features: {feature_names}\n")

# 80/20 train-test split, keeping class balance with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features so all models get fair input
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# -----------------------------------------------------------
# Helper function to get train and test metrics for any model
# -----------------------------------------------------------
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    avg = "weighted"

    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    def get_metrics(y_true, y_pred):
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
            "F1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
        }

    tr = get_metrics(y_tr, y_tr_pred)
    te = get_metrics(y_te, y_te_pred)

    return {
        "Model":           name,
        "Train Acc":       round(tr["Accuracy"],  4),
        "Test  Acc":       round(te["Accuracy"],  4),
        "Train Precision": round(tr["Precision"], 4),
        "Test  Precision": round(te["Precision"], 4),
        "Train Recall":    round(tr["Recall"],    4),
        "Test  Recall":    round(te["Recall"],    4),
        "Train F1":        round(tr["F1"],        4),
        "Test  F1":        round(te["F1"],        4),
    }


# -----------------------------------------------------------
# A2 - Hyperparameter tuning with RandomizedSearchCV
# -----------------------------------------------------------
print("Tuning hyperparameters with RandomizedSearchCV...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perceptron tuning
perceptron_params = {
    "penalty":  [None, "l2", "l1", "elasticnet"],
    "alpha":    [1e-4, 1e-3, 1e-2, 0.1],
    "max_iter": [500, 1000, 2000],
    "eta0":     [0.1, 0.5, 1.0],
}
perceptron_rs = RandomizedSearchCV(
    Perceptron(random_state=42), perceptron_params,
    n_iter=20, cv=cv, scoring="accuracy", random_state=42, n_jobs=-1
)
perceptron_rs.fit(X_train_sc, y_train)
print(f"Perceptron best params: {perceptron_rs.best_params_}")
print(f"Perceptron best CV accuracy: {perceptron_rs.best_score_:.4f}\n")

# Random Forest tuning
rf_params = {
    "n_estimators":      [50, 100, 200, 300],
    "max_depth":         [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features":      ["sqrt", "log2"],
}
rf_rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), rf_params,
    n_iter=20, cv=cv, scoring="accuracy", random_state=42, n_jobs=-1
)
rf_rs.fit(X_train_sc, y_train)
print(f"Random Forest best params: {rf_rs.best_params_}")
print(f"Random Forest best CV accuracy: {rf_rs.best_score_:.4f}\n")

# SVM tuning
svm_params = {
    "C":      [0.01, 0.1, 1, 10, 100],
    "kernel": ["rbf", "linear", "poly"],
    "gamma":  ["scale", "auto"],
}
svm_rs = RandomizedSearchCV(
    SVC(probability=True, random_state=42), svm_params,
    n_iter=20, cv=cv, scoring="accuracy", random_state=42, n_jobs=-1
)
svm_rs.fit(X_train_sc, y_train)
print(f"SVM best params: {svm_rs.best_params_}")
print(f"SVM best CV accuracy: {svm_rs.best_score_:.4f}\n")


# -----------------------------------------------------------
# A3 - Train all classifiers and compare results
# -----------------------------------------------------------
print("Training all classifiers...\n")

classifiers = {
    "Perceptron":     perceptron_rs.best_estimator_,
    "SVM":            svm_rs.best_estimator_,
    "Decision Tree":  DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":  rf_rs.best_estimator_,
    "AdaBoost":       AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boost": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Naive Bayes":    GaussianNB(),
    "MLP":            MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}

if XGBOOST_OK:
    classifiers["XGBoost"] = XGBClassifier(
        n_estimators=100, use_label_encoder=False,
        eval_metric="mlogloss", random_state=42
    )
if CATBOOST_OK:
    classifiers["CatBoost"] = CatBoostClassifier(iterations=100, verbose=0, random_seed=42)

results = []
trained_models = {}

for name, clf in classifiers.items():
    clf.fit(X_train_sc, y_train)
    trained_models[name] = clf
    row = evaluate(name, clf, X_train_sc, y_train, X_test_sc, y_test)
    results.append(row)
    print(f"  Done: {name}")

# Print the results table
df_results = pd.DataFrame(results)
df_results.set_index("Model", inplace=True)
print("\nResults (Train vs Test):")
print(df_results.to_string())
print()

# Bar chart comparing train and test accuracy across all models
fig, ax = plt.subplots(figsize=(12, 5))
df_results[["Train Acc", "Test  Acc"]].plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black")
ax.set_title("Train vs Test Accuracy - All Classifiers", fontsize=14)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Classifier")
ax.set_ylim(0, 1.1)
ax.legend(["Train Accuracy", "Test Accuracy"])
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("classifier_comparison.png", dpi=150)
plt.show()

# Confusion matrix for the best performing model
best_model_name = df_results["Test  Acc"].idxmax()
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=150)
plt.show()

print(f"\nClassification Report for {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=target_names))


# -----------------------------------------------------------
# 5-Fold Cross-Validation to check how stable each model is
# -----------------------------------------------------------
print("Running 5-fold cross-validation...\n")

cv_scores = {}
for name, clf in trained_models.items():
    scores = cross_val_score(clf, X_train_sc, y_train, cv=cv, scoring="accuracy")
    cv_scores[name] = scores
    print(f"  {name:<18} | mean={scores.mean():.4f}  std={scores.std():.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.boxplot(cv_scores.values(), labels=cv_scores.keys(), patch_artist=True)
ax.set_title("5-Fold CV Accuracy per Classifier", fontsize=13)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Classifier")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("cv_boxplot.png", dpi=150)
plt.show()


# -----------------------------------------------------------
# O1 - SHAP: see which features matter most to the model
# -----------------------------------------------------------
if SHAP_OK:
    print("\nRunning SHAP analysis on Random Forest...\n")

    rf_best = trained_models["Random Forest"]
    explainer = shap.TreeExplainer(rf_best)
    shap_values = explainer.shap_values(X_test_sc)

    # Summary plot showing feature impact across all classes
    plt.figure()
    shap.summary_plot(
        shap_values, X_test_sc,
        feature_names=feature_names,
        class_names=list(target_names),
        show=False
    )
    plt.title("SHAP Summary Plot - Random Forest")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Bar chart of feature importance for the first class
    plt.figure()
    shap.summary_plot(
        shap_values[0], X_test_sc,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f"SHAP Feature Importance - Class: {target_names[0]}")
    plt.tight_layout()
    plt.savefig("shap_bar_class0.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Force plot explaining one single prediction
    sample_idx = 0
    shap.initjs()
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][sample_idx],
        X_test_sc[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot - Test Sample {sample_idx}")
    plt.tight_layout()
    plt.savefig("shap_force_plot.png", dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("SHAP not installed. Run: pip install shap")


# -----------------------------------------------------------
# O2 - LIME: explain individual predictions locally
# -----------------------------------------------------------
if LIME_OK:
    print("\nRunning LIME explanations...\n")

    rf_best = trained_models["Random Forest"]

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_sc,
        feature_names=feature_names,
        class_names=list(target_names),
        discretize_continuous=True,
        mode="classification",
        random_state=42
    )

    # Explain 3 test samples
    for idx in [0, 5, 10]:
        exp = lime_explainer.explain_instance(
            data_row=X_test_sc[idx],
            predict_fn=rf_best.predict_proba,
            num_features=len(feature_names),
            top_labels=1
        )

        html_path = f"lime_explanation_sample_{idx}.html"
        exp.save_to_file(html_path)
        print(f"Saved: {html_path}  (true label = {target_names[y_test.iloc[idx]]})")

        fig = exp.as_pyplot_figure(label=exp.top_labels[0])
        plt.title(f"LIME - Sample {idx} (True: {target_names[y_test.iloc[idx]]})")
        plt.tight_layout()
        plt.savefig(f"lime_explanation_sample_{idx}.png", dpi=150, bbox_inches="tight")
        plt.show()

else:
    print("LIME not installed. Run: pip install lime")


# -----------------------------------------------------------
# Final summary sorted by test accuracy
# -----------------------------------------------------------
print("\nFinal Model Rankings (by Test Accuracy):")
summary = df_results[["Train Acc", "Test  Acc", "Test  F1"]].sort_values("Test  Acc", ascending=False)
print(summary.to_string())
print("\nDone!")
