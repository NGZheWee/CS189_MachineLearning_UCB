import numpy as np
import numpy.linalg as LA
import pickle
from PIL import Image

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = pickle.load(open('x_train.p', 'rb'), encoding='latin1')
    y_train = pickle.load(open('y_train.p', 'rb'), encoding='latin1')
    x_test = pickle.load(open('x_test.p', 'rb'), encoding='latin1')
    y_test = pickle.load(open('y_test.p', 'rb'), encoding='latin1')
    return x_train, y_train, x_test, y_test

def visualize_data(images: np.ndarray, controls: np.ndarray) -> None:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).
        controls (ndarray): control label array of size (n, 3).
    """
    # Convert images to uint8 for proper visualization
    images = images.astype(np.uint8)

    # Select the 0th, 10th, and 20th images
    indices_to_visualize = [0, 10, 20]

    # Visualize the selected images
    for idx in indices_to_visualize:
        img = images[idx]
        control = controls[idx]

        # Convert the numpy image array to a PIL image and display it
        image_to_show = Image.fromarray(img)
        image_to_show.show()

        # Print the control vector corresponding to this image
        print(f"Control vector for image {idx}: {control}")


def standardize_images(images: np.ndarray) -> np.ndarray:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).

    Returns:
        standardized_images (ndarray): standardized image array with values in range [-1, 1]
    """
    # Standardize pixel values to the range [-1, 1]
    standardized_images = (images / 255.0) * 2 - 1
    return standardized_images

def compute_data_matrix(images: np.ndarray, controls: np.ndarray, standardize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).
        controls (ndarray): control label array of size (n, 3).
        standardize (bool): boolean flag that specifies whether the images should be standardized or not

    Returns:
        X (ndarray): input array of size (n, 2700) where each row is the flattened image images[i]
        Y (ndarray): label array of size (n, 3) where row i corresponds to the control for X[i]
    """
    # TODO: Your code here!
    # Flatten each image into a single row vector (30 * 30 * 3 = 2700)
    X = images.reshape(images.shape[0], -1)

    # The controls (U) remain as they are (size n x 3)
    U = controls

    return X, U

def ridge_regression(X: np.ndarray, Y: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).
        lmbda (float): ridge regression regularization term

    Returns:
        pi (ndarray): learned policy of size (2700, 3)
    """
    # TODO: Your code here!
    n_features = X.shape[1]  # 2700
    identity_matrix = np.eye(n_features)  # Identity matrix for regularization

    # Ridge regression formula: pi = (X^T X + lambda I)^-1 X^T U
    pi = LA.inv(X.T @ X + lmbda * identity_matrix) @ X.T @ Y
    return pi

def ordinary_least_squares(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).

    Returns:
        pi (ndarray): learned policy of size (2700, 3)
    """ 
    # TODO: Your code here!
    try:
        pi = LA.inv(X.T @ X) @ X.T @ Y
    except LA.LinAlgError as e:
        print("Linear algebra error during OLS: ", e)
        pi = None
    return pi

def measure_error(X: np.ndarray, Y: np.ndarray, pi: np.ndarray) -> float:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).
        pi (ndarray): learned policy of size (2700, 3)

    Returns:
        error (float): the mean Euclidean distance error across all n samples
    """
    # TODO: Your code here!
    """
    n = X.shape[0]
    total_error = 0.0

    for i in range(n):
        predicted_u = X[i] @ pi  # Predicted control using pi
        true_u = U[i]  # True control
        total_error += np.linalg.norm(predicted_u - true_u) ** 2  # Squared Euclidean distance

    # Average the total error over all samples
    mean_error = total_error / n
    return mean_error
    """
    n = X.shape[0]
    total_error = 0.0
    for i in range(n):
        predicted_u = X[i] @ pi  # Predicted control using pi
        true_u = Y[i]  # True control
        total_error += np.linalg.norm(predicted_u - true_u) ** 2  # Squared Euclidean distance
    mean_error = total_error / n  # Average the total error over all samples
    return mean_error

def compute_condition_number(X: np.ndarray, lmbda: float) -> float:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        lmbda (float): ridge regression regularization term

    Returns:
        kappa (float): condition number of the input array with the given lambda
    """
    # TODO: Your code here!
    # Compute X^T X
    XtX = X.T @ X

    # Add regularization term (lambda * I)
    regularized_matrix = XtX + lmbda * np.eye(X.shape[1])

    # Compute singular values using SVD
    singular_values = LA.svd(regularized_matrix, compute_uv=False)

    # Condition number is the ratio of the maximum to minimum singular value
    condition_number = singular_values.max() / singular_values.min()
    return condition_number

if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_data()
    print("successfully loaded the training and testing data")

    LAMBDA = [0.1, 1.0, 10.0, 100.0, 1000.0]

    # TODO: Your code here!

    visualize_data(x_train, y_train)


    """    
    # Step 1: Prepare the data matrices
    X, U = compute_data_matrix(x_train, y_train)

    # Step 2: Perform ordinary least squares to compute pi
    pi = ordinary_least_squares(X, U)

    if pi is not None:
        print("Computed OLS solution for pi.")
    else:
        print("OLS solution could not be computed due to a singular matrix.")
    """

    """
    # Step 1: Prepare the data matrices
    X, U = compute_data_matrix(x_train, y_train)

    # Step 2: Perform ridge regression for each value of lambda
    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    for lmbda in lambdas:
        pi = ridge_regression(X, U, lmbda)
        training_error = measure_error(X, U, pi)
        print(f"Lambda: {lmbda}, Training Error: {training_error}")
    """

    """
    # Step 1: Standardize the training images
    x_train_standardized = standardize_images(x_train)

    # Step 2: Prepare the data matrices
    X, U = compute_data_matrix(x_train_standardized, y_train)

    # Step 3: Perform ridge regression for each value of lambda
    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    for lmbda in lambdas:
        pi = ridge_regression(X, U, lmbda)
        training_error = measure_error(X, U, pi)
        print(f"Lambda: {lmbda}, Training Error: {training_error}")
    """

    """
    # Non-standardized data
    X_train_non_standardized, U_train = compute_data_matrix(x_train, y_train)
    X_test_non_standardized, U_test = compute_data_matrix(x_test, y_test)

    # Standardized data
    x_train_standardized = standardize_images(x_train)
    x_test_standardized = standardize_images(x_test)
    X_train_standardized, _ = compute_data_matrix(x_train_standardized, y_train)
    X_test_standardized, _ = compute_data_matrix(x_test_standardized, y_test)

    # Regularization values
    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    print("Evaluating performance on test data:")

    for lmbda in lambdas:
        # Ridge regression for non-standardized data
        pi_non_standardized = ridge_regression(X_train_non_standardized, U_train, lmbda)
        test_error_non_standardized = measure_error(X_test_non_standardized, U_test, pi_non_standardized)

        # Ridge regression for standardized data
        pi_standardized = ridge_regression(X_train_standardized, U_train, lmbda)
        test_error_standardized = measure_error(X_test_standardized, U_test, pi_standardized)

        # Report errors
        print(f"Lambda: {lmbda}")
        print(f"  Non-standardized Test Error: {test_error_non_standardized}")
        print(f"  Standardized Test Error: {test_error_standardized}")
    """

    # Regularization value
    lmbda = 100.0

    # Compute condition number for non-standardized data
    X_train_non_standardized, _ = compute_data_matrix(x_train, y_train)
    condition_number_non_standardized = compute_condition_number(X_train_non_standardized, lmbda)
    print(f"Condition number (non-standardized): {condition_number_non_standardized}")

    # Compute condition number for standardized data
    x_train_standardized = standardize_images(x_train)
    X_train_standardized, _ = compute_data_matrix(x_train_standardized, y_train)
    condition_number_standardized = compute_condition_number(X_train_standardized, lmbda)
    print(f"Condition number (standardized): {condition_number_standardized}")