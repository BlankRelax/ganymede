import numpy as np

from data_generator import data_generator

y,x=data_generator().generate_multivariate_linear_data(number_of_variables=1,m=np.array([3]),c=5)

print(x,y.shape)