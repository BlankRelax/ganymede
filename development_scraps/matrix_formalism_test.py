import numpy as np
from data_generator import data_generator
from LinearRegressor import LinearRegressor

y,x=data_generator().generate_multivariate_linear_data(number_of_variables=3,m=np.array([5,4,6]),c=5,
                                                       length=10000, noise=False)
y_test,x_test=data_generator().generate_multivariate_linear_data(number_of_variables=3,m=np.array([5,4,6]),c=5,
                                                                 length=10000, noise=False)
lr=LinearRegressor(learning_rate=0.25, error_threshold=0.0001,tolerance=10,weights=[1,1,1,1])
lr.fit(y=y, x=x,optimizer='SGD')
#lr.fit(y=np.array(([3,4],)).T, x=np.array(([2,3],)).T)
print(lr.beta)
_,score=lr.predict(y=y_test, x=x_test)
print(score)



