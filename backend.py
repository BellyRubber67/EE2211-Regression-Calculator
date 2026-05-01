import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor

def LinRegression(X, y):
    poly = PolynomialFeatures(1)
    X = poly.fit_transform(X)
    w = np.linalg.inv(X.T@X) @ X.T @ y
    return w

def PolyRegression(X, y, order, reg_factor=0):    
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    rows, cols = P.shape
    if rows <= cols:
        w = P.T @ np.linalg.inv(P @ P.T + reg_factor * np.identity(rows)) @ y
    else:
        w = np.linalg.inv(P.T @ P + reg_factor * np.identity(cols)) @ P.T @ y
    
    return w

def predict(X, w, order=1, format="raw"):
    poly = PolynomialFeatures(order)
    X = poly.fit_transform(X)
    y = X @ w
    if format == "raw":
        return y
    if format == "argmax":
        return np.array([[1 if j == max(i) else 0 for j in i] for i in y])
    if format == "sign":
        return np.sign(y)
    
def regressionTree(X, y, depth=1):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X, y)
    return tree

def regressionTree_mse(X, y, depth=0):
    """
    Fit a regression tree at a given depth and return overall MSE on (X, y).

    Depth convention:
      - depth=0: root node only (constant prediction = mean(y))
      - depth=1: nodes up to depth 1, etc. (sklearn's max_depth)
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    # Normalize y shape to avoid broadcasting bugs:
    # - (n, 1) -> (n,)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.reshape(-1)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if depth < 0:
        raise ValueError("depth must be >= 0")

    # Weighted-by-leaf MSE (equivalently, overall average squared error)
    # depth=0: root node only => one leaf containing all samples
    if depth == 0:
        mu = np.mean(y_arr, axis=0)
        return float(np.mean((y_arr - mu) ** 2))

    tree = regressionTree(X_arr, y_arr, depth=depth)

    # Compute MSE per leaf, then weight by leaf sample count / total samples
    leaf_id = tree.apply(X_arr)
    unique_leaves, counts = np.unique(leaf_id, return_counts=True)
    n_total = float(X_arr.shape[0])

    mse_total = 0.0
    for lid, cnt in zip(unique_leaves, counts):
        mask = leaf_id == lid
        y_leaf = y_arr[mask]
        mu = np.mean(y_leaf, axis=0)
        mse_leaf = float(np.mean((y_leaf - mu) ** 2))
        mse_total += (cnt / n_total) * mse_leaf

    return float(mse_total)

def manual_split_tree(X, y, split_x, feature_index=0):
    """
    Manually split data at a chosen x value (a regression-tree stump split).

    Split rule:
      left:  x <= split_x
      right: x >  split_x

    Predictions:
      left region  -> mean(y_left)
      right region -> mean(y_right)

    Returns a dict containing the split subsets and weighted MSE.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if X_arr.ndim == 1:
        x = X_arr.reshape(-1)
    else:
        x = X_arr[:, feature_index].reshape(-1)

    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_flat = y_arr.reshape(-1)
    else:
        y_flat = y_arr.reshape(-1)

    if x.shape[0] != y_flat.shape[0]:
        m = min(x.shape[0], y_flat.shape[0])
        x = x[:m]
        y_flat = y_flat[:m]

    left_mask = x <= split_x
    right_mask = ~left_mask

    y_left = y_flat[left_mask]
    y_right = y_flat[right_mask]

    # Handle empty branch(es): fall back to global mean for that side
    y_mean = float(np.mean(y_flat)) if y_flat.size else 0.0
    left_mean = float(np.mean(y_left)) if y_left.size else y_mean
    right_mean = float(np.mean(y_right)) if y_right.size else y_mean

    mse_left = float(np.mean((y_left - left_mean) ** 2)) if y_left.size else 0.0
    mse_right = float(np.mean((y_right - right_mean) ** 2)) if y_right.size else 0.0

    n = float(y_flat.size) if y_flat.size else 1.0
    mse_total = (y_left.size / n) * mse_left + (y_right.size / n) * mse_right

    return {
        "split_x": float(split_x),
        "n_total": int(y_flat.size),
        "n_left": int(y_left.size),
        "n_right": int(y_right.size),
        "left_mean": left_mean,
        "right_mean": right_mean,
        "mse_left": mse_left,
        "mse_right": mse_right,
        "mse_total": float(mse_total),
        "left": {"x": x[left_mask], "y": y_left},
        "right": {"x": x[right_mask], "y": y_right},
    }

def manual_split_predict(X, split_x, left_mean, right_mean, feature_index=0):
    """Predict using a manual split stump (means on each side)."""
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        x = X_arr.reshape(-1)
    else:
        x = X_arr[:, feature_index].reshape(-1)
    preds = np.where(x <= split_x, left_mean, right_mean)
    return preds

def manual_split_mse(X, y, split_x, feature_index=0):
    """
    Overall weighted MSE for a manual split stump at split_x.

    This is the same mse_total returned by manual_split_tree, exposed as a
    convenience function.
    """
    info = manual_split_tree(X, y, split_x, feature_index=feature_index)
    return float(info["mse_total"])

if __name__ == "__main__":
    X = np.array([0.2, 0.7, 1.8, 2.2, 3.7, 4.1, 4.5, 5.1, 6.3, 7.4])
    y = np.array([2.1, 1.5, 5.8, 6.1, 9.1, 9.5, 9.8, 12.7, 13.8, 15.9])

    print(regressionTree_mse(X,y, depth=0))
    print(manual_split_mse(X,y, split_x=3, feature_index=0))