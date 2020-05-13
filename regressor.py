import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error



ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators = 10, max_features = 32, random_state = 0),
        "K-nn": KNeighborsRegressor(),
        "Linear regression": LinearRegression,
        "Ridge": RidgeCV()
        }

for i in range(8):
    errors={}
    x=[]
    y=[]
    with open("test_func"+str(i+1) + "_x", "r") as f:
        lines = f.readlines()
        for line in lines:
            x.append(float(line.rstrip()))

    with open("test_func"+str(i+1) + "_y", "r") as f:
        lines = f.readlines()
        for line in lines:
            y.append(float(line.rstrip()))

    for name, estimator in ESTIMATORS.items():
        estimator.fit(x, y)
        y_predicted = estimator.predict(x)
        mae = mean_absolute_error(y, y_predicted)
        errors[name] = mae

    print("test_func"+str(i+1))
    print(errors)



