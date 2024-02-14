import numpy as np

from LinearRegressor import LinearRegressor

lr=LinearRegressor(learning_rate=0.01, error_threshold=0.001,tolerance=10,weights=[0,1.5])
lr.fit(y=np.array(([2,3],)).T, x=np.array(([2,3],)).T)
print(lr.beta)
