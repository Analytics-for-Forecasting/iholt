# interval Holt’s exponential smoothing method

Interval Holt’s exponential smoothing is a method which implements Holt-winter algorithm with interval-valued inputs and interval-valued outputs. This package is based on the paper, [Holt's exponential smoothing and neural network models for forecasting interval-valued time series](https://www.sciencedirect.com/science/article/pii/S0169207010000506), [FAT de Carvalho](https://scholar.google.com/citations?user=7t7NjEUAAAAJ&hl=en&oi=sra)



## Usage

```python
import numpy as np
from iholt import Holt_model
from scipy.optimize import minimize

# Construct the synthetic data
# 2 * len_series
# dimension 1: lower bound
# dimension 2: upper bound
s = np.sin(np.arange(0, 9, 0.01))
s = np.vstack((s, s+0.5))

# Build model
holt_model = Holt_model(s)

# Optimize
bnds = [[0, 1]] * 8 # each parameter is in range [0, 1]
x0 = np.random.rand(8)   # initialize parameters [a11, a12, a21, a22, b11, b12, b21, b22]
result = minimize(holt_model.fun, x0, method='L-BFGS-B', bounds=bnds)

print(result)
```

