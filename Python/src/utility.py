import numpy as np

def total_poisson_dev(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    nlogn = np.where(y == 0, 0, y * np.log(y / y_pred))
    dev = 2 * (nlogn - (y - y_pred))
    return dev.sum()
