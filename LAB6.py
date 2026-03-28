import pandas as pd
import math
import pandas as pd

def calculate_entropy(data, target_col):
    total = len(data)
    counts = {}

    for i in range(total):
        value = data[target_col][i]
        if value not in counts:
            counts[value] = 0
        counts[value] += 1

    entropy = 0
    for key in counts:
        p = counts[key] / total
        entropy = entropy - (p * math.log(p, 2))

    return entropy


def calculate_gini(data, target_col):
    total = len(data)
    counts = {}

    for i in range(total):
        value = data[target_col][i]
        if value not in counts:
            counts[value] = 0
        counts[value] += 1

    gini = 1
    for key in counts:
        p = counts[key] / total
        gini = gini - (p * p)

    return gini


def equal_width_binning(data, column, bins):
    min_val = min(data[column])
    max_val = max(data[column])

    width = (max_val - min_val) / bins

    new_column = []

    for i in range(len(data)):
        value = data[column][i]

        bin_index = int((value - min_val) / width)

        if bin_index >= bins:
            bin_index = bins - 1

        new_column.append("bin_" + str(bin_index))

    return new_column


def information_gain(data, feature, target_col):
    total_entropy = calculate_entropy(data, target_col)

    total = len(data)
    feature_values = []

    for i in range(total):
        val = data[feature][i]
        if val not in feature_values:
            feature_values.append(val)

    weighted_entropy = 0

    for val in feature_values:
        subset_rows = []

        for i in range(total):
            if data[feature][i] == val:
                subset_rows.append(i)

        subset_data = data.iloc[subset_rows].reset_index(drop=True)

        weight = len(subset_data) / total
        subset_entropy = calculate_entropy(subset_data, target_col)

        weighted_entropy += weight * subset_entropy

    gain = total_entropy - weighted_entropy
    return gain


def best_feature(data, features, target_col):
    best = None
    best_gain = -1

    for feature in features:
        gain = information_gain(data, feature, target_col)

        if gain > best_gain:
            best_gain = gain
            best = feature

    return best


def build_tree(data, features, target_col):

    first = data[target_col][0]
    same = True

    for i in range(len(data)):
        if data[target_col][i] != first:
            same = False
            break

    if same:
        return first

    if len(features) == 0:
        return first

    root = best_feature(data, features, target_col)

    tree = {}
    tree[root] = {}

    values = []
    for i in range(len(data)):
        val = data[root][i]
        if val not in values:
            values.append(val)

    for val in values:
        subset_rows = []

        for i in range(len(data)):
            if data[root][i] == val:
                subset_rows.append(i)

        
        subset_data = data.iloc[subset_rows].reset_index(drop=True)

        if len(subset_data) == 0:
            tree[root][val] = first
        else:
            new_features = []
            for f in features:
                if f != root:
                    new_features.append(f)

            subtree = build_tree(subset_data, new_features, target_col)
            tree[root][val] = subtree

    return tree


def main():
    df = pd.read_csv("Gender_Classification_Data.csv")

    print("Original Data:")
    print(df.head())

    target_col = "gender"

    for col in df.columns:
        if col != target_col:
            df[col] = equal_width_binning(df, col, 4)

    print("\nAfter Binning:")
    print(df.head())

    ent = calculate_entropy(df, target_col)
    print("\nEntropy:", ent)

    gini = calculate_gini(df, target_col)
    print("Gini:", gini)

    features = []
    for col in df.columns:
        if col != target_col:
            features.append(col)

    best = best_feature(df, features, target_col)
    print("\nBest Feature:", best)

    tree = build_tree(df, features, target_col)
    print("\nDecision Tree:")
    print(tree)


if __name__ == "__main__":
    main()
