from scipy.optimize import minimize
import numpy as np

def get_Lt(a, Xt, Lt_1, Tt_1):
    return np.dot(a, Xt) + np.dot((np.diag([1, 1]) - a), Lt_1 + Tt_1)

def get_Tt(b, Lt, Lt_1, Tt_1):
    return np.dot(b, Lt - Lt_1) + np.dot((np.diag([1, 1]) - b), Tt_1)

class Holt_model(object):
    def __init__(self, s):
        self.s = s
    def fun(self, x):
        a = x[:4].reshape((2, 2))
        b = x[4:].reshape((2, 2))

        s = self.s
        Lt_1 = s[:, 1:2]
        Tt_1 = s[:, 1:2] - s[:, 0:1]
        e = np.sum((s[:, 2:3] - Lt_1 - Tt_1) ** 2)
        for i in range(len(self.s[0]) - 3):
            Lt = get_Lt(a, s[:, i+2:i+3], Lt_1, Tt_1)
            Tt = get_Tt(b, Lt, Lt_1, Tt_1)
            e = e + np.sum((s[:, i+3:i+4] - Lt - Tt) ** 2)
        return e

# if __name__ == "__main__":
#     # Construct the synthetic data
#     s = np.sin(np.arange(0, 9, 0.01))
#     s = np.vstack((s, s+0.5))

#     # Build model
#     holt_model = Holt_model(s)

#     # Optimize
#     bnds = [[0, 1]] * 8
#     x0 = np.random.rand(8)   # Parameters [a11, a12, a21, a22, b11, b12, b21, b22]
#     result = minimize(holt_model.fun, x0, method='L-BFGS-B', bounds=bnds)

#     print(result)