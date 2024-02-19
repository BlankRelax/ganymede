from MachineLearning.LinearRegressor import LinearRegressor
from data_generator import data_generator
import numpy as np
import matplotlib.pyplot as plt

lr=LinearRegressor(learning_rate=0.01, error_threshold=0.0001,tolerance=10, initial_weights = [5,1] )
y,x=data_generator().generate_multivariate_linear_data(number_of_variables=1, m=np.array([9]),c=0,noise=True)
lr.fit(y=y, x=x)
print(lr.m_i, lr.c,)
y_true, x_test = data_generator().generate_multivariate_linear_data(number_of_variables=1, m=np.array([7]),c=5,noise=True)
y_hat=lr.predict(x=x_test)
print(sum(y_hat-y_true)/200)


def _plot_line(y, x,y_hat):
    plt.scatter(x, y, c='r', label='true data')
    plt.plot(x,y_hat, c='c', label='regression model')
    plt.title('Linear regression prediction against true data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.show()
    plt.clf()

y_hat = []
for n in range(len(y)):
    inner_sum = 0
    for i in range(1):
        inner_sum += lr.m_i[i] * x[n, i] + lr.c
        y_hat.append(inner_sum)

_plot_line(y=y,x=x,y_hat=y_hat)









