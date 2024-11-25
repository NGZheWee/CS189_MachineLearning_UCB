import numpy as np
import pandas as pd
import scipy.io
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import random

random.seed(246810)
np.random.seed(246810)
eps = 1e-5  # A small number

# DecisionTree Class
class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None
        self.split_idx, self.thresh = None, None
        self.pred = None

    @staticmethod
    def entropy(y):
        y = y.astype(int)
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    @staticmethod
    def information_gain(X_column, y, thresh):
        parent_entropy = DecisionTree.entropy(y)
        left_indices = X_column < thresh
        right_indices = ~left_indices

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left = np.sum(left_indices)
        n_right = n - n_left

        e_left = DecisionTree.entropy(y[left_indices])
        e_right = DecisionTree.entropy(y[right_indices])

        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def split(self, X, y, feature_idx, thresh):
        left_indices = X[:, feature_idx] < thresh
        right_indices = ~left_indices
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    def fit(self, X, y, depth=0):
        y = y.astype(int)
        if depth >= self.max_depth or len(set(y)) == 1:
            self.pred = Counter(y).most_common(1)[0][0]
            return

        best_gain = 0
        best_split = None

        n_features = X.shape[1]
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for thresh in thresholds:
                gain = self.information_gain(X_column, y, thresh)
                if gain > best_gain + eps:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'thresh': thresh
                    }

        if best_gain == 0 or best_split is None:
            self.pred = Counter(y).most_common(1)[0][0]
            return

        self.split_idx = best_split['feature_idx']
        self.thresh = best_split['thresh']

        X_left, y_left, X_right, y_right = self.split(X, y, self.split_idx, self.thresh)

        if len(y_left) == 0 or len(y_right) == 0:
            self.pred = Counter(y).most_common(1)[0][0]
            return

        self.left = DecisionTree(self.max_depth, self.features)
        self.left.fit(X_left, y_left, depth + 1)

        self.right = DecisionTree(self.max_depth, self.features)
        self.right.fit(X_right, y_right, depth + 1)

    def predict(self, X):
        y_pred = np.array([self._predict_single(x) for x in X])
        return y_pred

    def _predict_single(self, x):
        if self.pred is not None:
            return self.pred
        else:
            if x[self.split_idx] < self.thresh:
                return self.left._predict_single(x)
            else:
                return self.right._predict_single(x)

    def trace(self, x):
        """
        Trace the path for a single data point x.
        Returns a list of (feature, threshold, direction, value).
        """
        path = []
        node = self
        while node.pred is None:
            feature_name = self.features[node.split_idx] if self.features else f"Feature {node.split_idx}"
            if x[node.split_idx] < node.thresh:
                direction = '<'
                path.append((feature_name, node.thresh, direction, x[node.split_idx]))
                node = node.left
            else:
                direction = '>='
                path.append((feature_name, node.thresh, direction, x[node.split_idx]))
                node = node.right
        return path, node.pred

# BaggedTrees Class
class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n_estimators=200):
        self.params = params if params is not None else {}
        self.n_estimators = n_estimators
        self.decision_trees = []

    def fit(self, X, y):
        y = y.astype(int)
        self.classes_ = np.unique(y)
        self.decision_trees = []
        n_samples = X.shape[0]
        for i in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            tree = DecisionTreeClassifier(random_state=i, **self.params)
            tree.fit(X_bootstrap, y_bootstrap)
            self.decision_trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.decision_trees])
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions.astype(int))
        return y_pred

# RandomForest Class
class RandomForest(BaggedTrees):
    def __init__(self, params=None, n_estimators=200, max_features='sqrt'):
        self.params = params if params is not None else {}
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.params['max_features'] = self.max_features
        super().__init__(params=self.params, n_estimators=self.n_estimators)

# Preprocess Function
def preprocess(data, onehot_cols=[]):
    data = data.copy()
    for col in onehot_cols:
        one_hot = pd.get_dummies(data[col], prefix=col)
        data = data.drop(col, axis=1)
        data = data.join(one_hot)
    data = data.fillna(0)
    return data

# Generate Submission Function
def generate_submission(testing_data, predictions, dataset="titanic"):
    assert dataset in ["titanic", "spam"], f"dataset should be either 'titanic' or 'spam'"
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(f'predictions_{dataset}.csv', index_label='Id')

# Main Function for Q5.5
def main_q55():
    """
    Q5.5
    """
    # Load spam data
    print("Loading Spam Dataset")
    path_train = 'datasets/spam_data/spam_data.mat'
    data = scipy.io.loadmat(path_train)

    # Define custom feature names for the spam dataset
    features = [
        "pain", "private", "bank", "money", "drug", "spam", "prescription",
        "creative", "height", "featured", "differ", "width", "other",
        "energy", "business", "message", "volumes", "revision", "path",
        "meter", "memo", "planning", "pleased", "record", "out",
        "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
        "square_bracket", "ampersand"
    ]

    X = pd.DataFrame(data['training_data'])
    y = np.squeeze(data['training_labels']).astype(int)

    # Split into training and validation sets
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X.values, y, test_size=0.2, random_state=42
    )

    # Task 1: Decision Tree Paths for Specific Data Points
    print("\nTask 1: Decision Tree Paths for Specific Data Points")
    dt = DecisionTree(max_depth=5, feature_labels=features)
    dt.fit(X_train_full, y_train_full)

    # Choose one spam and one ham example
    spam_indices = np.where(y_train_full == 1)[0]
    ham_indices = np.where(y_train_full == 0)[0]

    # Select the first example from each class
    spam_example = X_train_full[spam_indices[0]]
    ham_example = X_train_full[ham_indices[0]]

    # Trace the paths
    spam_path, spam_pred = dt.trace(spam_example)
    ham_path, ham_pred = dt.trace(ham_example)

    print("\nSpam Email Classification Path:")
    for idx, (feature, thresh, direction, value) in enumerate(spam_path):
        print(f"({idx+1}) {feature} {direction} {thresh} (Value: {value})")
    print(f"Therefore, this email was classified as {'spam' if spam_pred == 1 else 'ham'}.")

    print("\nHam Email Classification Path:")
    for idx, (feature, thresh, direction, value) in enumerate(ham_path):
        print(f"({idx+1}) {feature} {direction} {thresh} (Value: {value})")
    print(f"Therefore, this email was classified as {'spam' if ham_pred == 1 else 'ham'}.")

    # Task 2: Most Common Splits at Root Node in Random Forest
    print("\nTask 2: Most Common Splits at Root Node in Random Forest")
    params = {"max_depth": 5, "min_samples_leaf": 10}
    n_estimators = 100
    rf = RandomForest(params=params, n_estimators=n_estimators)
    rf.fit(X_train_full, y_train_full)

    root_splits = []
    for tree in rf.decision_trees:
        if hasattr(tree, 'tree_'):
            feature_idx = tree.tree_.feature[0]
            threshold = tree.tree_.threshold[0]
            if feature_idx != -2:  # -2 indicates leaf node
                root_splits.append((features[feature_idx], threshold))

    # Count the most common splits
    counter = Counter([split[0] for split in root_splits])
    most_common_splits = counter.most_common()

    print("Most Common Root Splits:")
    for feature, count in most_common_splits:
        print(f"Feature: {feature} ({count} trees)")

    # Task 3: Decision Tree Depth vs. Validation Accuracy
    print("\nTask 3: Decision Tree Depth vs. Validation Accuracy")
    max_depths = range(1, 41)
    train_accuracies = []
    val_accuracies = []

    # Use a fixed training/validation split
    X_train, X_val_split, y_train, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    for depth in max_depths:
        dt = DecisionTree(max_depth=depth)
        dt.fit(X_train, y_train)
        # Training accuracy
        y_pred_train = dt.predict(X_train)
        train_accuracy = np.mean(y_pred_train == y_train)
        train_accuracies.append(train_accuracy)
        # Validation accuracy
        y_pred_val = dt.predict(X_val_split)
        val_accuracy = np.mean(y_pred_val == y_val_split)
        val_accuracies.append(val_accuracy)

    # Plot validation accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, val_accuracies, label='Validation Accuracy', marker='o')
    plt.plot(max_depths, train_accuracies, label='Training Accuracy', marker='x')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Depth vs. Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('decision_tree_depth_accuracy.png')  # Save the plot to a file
    print("Plot saved as 'decision_tree_depth_accuracy.png'.")

    # Find the depth with the highest validation accuracy
    best_depth = max_depths[np.argmax(val_accuracies)]
    print(f"\nMaximum validation accuracy of {max(val_accuracies):.4f} achieved at depth = {best_depth}")

    print("\nObservation:")
    print("As the depth increases, the training accuracy generally improves, indicating that the model is fitting the training data better.")
    print("However, the validation accuracy may peak at a certain depth and then decrease, suggesting overfitting beyond that depth.")

# Main Execution
if __name__ == "__main__":
    # Uncomment the function corresponding to the question you want to run

    # For Q5.5
    main_q55()

    # For Q5.4
    # main_q54()

    # For Q5.1, Q5.2, Q5.6
    # main_q5126()
