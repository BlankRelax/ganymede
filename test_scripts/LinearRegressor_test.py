import numpy as np

from MachineLearning import LinearRegressor
from data_generator import data_generator

y,x = data_generator().generate_multivariate_linear_data(number_of_variables=2,
                                                       m=[3,4],
                                                       c=1,
                                                       length=500)

def test_initialise():


    lr=LinearRegressor(learning_rate=0.1,
                    error_threshold=2,
                    tolerance=5,
                    weights=[1,3,4])

    assert lr.is_fitted == False

def test_multi_dim_error():
    lr = LinearRegressor(learning_rate=0.1,
                         error_threshold=2,
                         tolerance=5,
                         weights=[1, 1])
    e=lr.multi_dim_error(y=np.array(([1,1],)).T,x=np.array(([0,1],[0,1])),func='MSE')
    assert e==0

def test_cfd():
    lr = LinearRegressor(learning_rate=0.1,
                         error_threshold=2,
                         tolerance=5,
                         weights=[1, 1])

    c_prime = lr.cost_function_derivative(y=np.array(([1,1],)).T,
                                          x=np.array(([1,0],[0,1]))
                                          )
    expected_value = np.array(([0,0],)).T

    assert np.array_equal(c_prime, expected_value)




