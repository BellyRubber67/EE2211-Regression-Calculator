import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

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

if __name__ == "__main__":
    X = np.array([
        [1,2,3],
        [4,0,6],
        [1,1,0],
        [0,1,2],
        [5,7,-2],
        [-1,4,0]
    ])
    y = np.array([
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,1,0],
        [0,0,1]
    ])

    print(LinRegression(X,y))