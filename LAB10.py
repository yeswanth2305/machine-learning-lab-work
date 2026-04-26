import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector

from lime.lime_tabular import LimeTabularExplainer


def load_data():
    data = pd.read_csv("Gender_Classification_Data.csv")
    return data


def preprocess(data):
    data = data.copy()

    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])

    X = data.drop('gender', axis=1)
    y = data['gender']

    return X, y, data



# A1: CORRELATION HEATMAP

def correlation_plot(data):
    corr = data.corr()

    plt.figure()
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Correlation Heatmap")
    plt.show()


# PCA VARIANCE GRAPH

def plot_pca_variance(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(cum_var, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.title("PCA Variance Explained")
    plt.grid()
    plt.show()


# PCA FUNCTION

def perform_pca(X, variance):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n = np.argmax(cum_var >= variance) + 1

    pca_final = PCA(n_components=n)
    X_new = pca_final.fit_transform(X_scaled)

    return X_new, n


# TRAIN MODEL

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return model, acc, X_train, X_test


# A4: FEATURE SELECTION

def feature_selection(X, y):
    model = LogisticRegression(max_iter=1000)

    sfs = SequentialFeatureSelector(model, n_features_to_select=2)
    sfs.fit(X, y)

    features = X.columns[sfs.get_support()]
    return features


# FEATURE PLOT

def plot_selected_features(features):
    values = [1] * len(features)

    plt.bar(features, values)
    plt.title("Selected Features (SFS)")
    plt.ylabel("Selected")
    plt.show()


# ACCURACY COMPARISON

def plot_accuracy(acc_99, acc_95, acc_sfs):
    methods = ['PCA 99%', 'PCA 95%', 'SFS']
    acc = [acc_99, acc_95, acc_sfs]

    plt.bar(methods, acc)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.show()


# A5: LIME

def lime_explain(model, X_train, X_test):
    explainer = LimeTabularExplainer(
        X_train,
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_test[0],
        model.predict_proba
    )

    return exp


if __name__ == "__main__":

    data = load_data()

    X, y, data_processed = preprocess(data)

    correlation_plot(data_processed)

    plot_pca_variance(X)

    # A2: PCA 99%
    X_99, comp_99 = perform_pca(X, 0.99)
    model_99, acc_99, Xtr_99, Xte_99 = train_model(X_99, y)

    # A3: PCA 95%
    X_95, comp_95 = perform_pca(X, 0.95)
    model_95, acc_95, Xtr_95, Xte_95 = train_model(X_95, y)

    # A4: SFS
    selected = feature_selection(X, y)
    plot_selected_features(selected)

    X_sfs = X[selected]
    model_sfs, acc_sfs, _, _ = train_model(X_sfs, y)

    plot_accuracy(acc_99, acc_95, acc_sfs)

    # A5: LIME
    lime_exp = lime_explain(model_99, Xtr_99, Xte_99)

    fig = lime_exp.as_pyplot_figure()
    plt.show()

  
    print("\nPCA 99% Components:", comp_99)
    print("Accuracy PCA 99%:", acc_99)

    print("\nPCA 95% Components:", comp_95)
    print("Accuracy PCA 95%:", acc_95)

    print("\nSelected Features:", list(selected))
    print("Accuracy SFS:", acc_sfs)
