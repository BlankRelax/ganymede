from LinearRegressor import LinearRegressor
from data_generator import data_generator
import numpy as np

lr=LinearRegressor(learning_rate=0.1, error_threshold=0.1,tolerance=3)
y,x = data_generator().generate_linear_data(start=0,stop=250,m=1,c=0)
lr.fit(y=y, x=x)




