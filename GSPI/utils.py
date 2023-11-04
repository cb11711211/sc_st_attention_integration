from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(pred, target):
    return np.sqrt(mean_squared_error(pred, target))
