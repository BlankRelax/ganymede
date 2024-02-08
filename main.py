from LinearRegressor import LinearRegressor
from data_generator import data_generator
import numpy as np
import matplotlib.pyplot as plt

lr=LinearRegressor(learning_rate=0.000001, error_threshold=0.1,tolerance=3)
y,x = data_generator().generate_linear_data(start=0,stop=100, noise=1,m=1,c=3)
y1,x1 = data_generator().generate_linear_data(start=0,stop=100, noise=1,m=1,c=3)
x_multi = np.append(x,x1)
lr.fit(y=y, x=(x_multi).reshape(50,2))
print(lr.m_i, lr.c)








