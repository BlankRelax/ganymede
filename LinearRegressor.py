# code for linear regressor
import  numpy as np
import matplotlib.pyplot as plt
from base.base import base_regressor
from typing import Literal



class LinearRegressor(base_regressor):

    def __init__(self, learning_rate, error_threshold, tolerance, weights:list):

        # hyper-parameters
        self.learning_rate=learning_rate
        self.error_threshold = error_threshold
        self.tolerance = tolerance

        # properties
        self._is_fitted: bool = False

        self.y_hat=[]

        # matrices
        self.beta = np.array((weights,)).T

    def multi_dim_error(self,y,x,func:Literal['MSE']):

        if func=='MSE':
            e_beta = y - np.matmul(x,self.beta)
            error = (1/self.N) * np.matmul(e_beta.T,e_beta)
        elif func=='RMSE':
            e_beta = y - np.matmul(x1=x, x2=self.beta)
            error = np.sqrt(x= (1 / self.N) * np.matmul(x1=e_beta.T, x2=e_beta))

        return error

    def fit(self, y,x):

        #method current supports 1-D Linear regression fit

        self.N = len(y) # number of observations

        if type(y)!=np.ndarray:
            raise TypeError('y is not a numpy array')
        if type(x)!=np.ndarray:
            raise TypeError('x is not a numpy array')

        if y.shape[0]!=x.shape[0]:
            raise ValueError('y does not have the same number of rows and x')

        x = np.insert(arr=x, obj=0, values=np.ones(shape=(self.N,)), axis=1)

        #LinearRegressor.multi_dim_gradient_descent(self,y,x)
        self._is_fitted=True
        return LinearRegressor.multi_dim_error(self,y=y,x=x,func='MSE')

    def predict(self,x:np.ndarray)->np.ndarray:
        y=[]

        if not self._is_fitted:
            raise PermissionError('you have cannot do this as you have not fit the regressor')
        if x.shape[1]!=len(self.m_i):
            raise ValueError('x has the wrong number of dimensions')

        for row_index in range(x.shape[0]):
            y.append(np.dot(a=self.m_i,b=x[row_index,:].T))

        return np.array(y)

    def _plot_line(self,y,x):
        plt.scatter(x,y,c='r', label='true data')
        plt.plot(x,self.y_hat, c='c',label='regression model')
        plt.title('Linear regression prediction against true data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='lower right')
        plt.show()
        plt.clf()


    def cost_function_derivative_bias(self, y, x):

        outer_sum=0
        for n in range(self.N):
            inner_sum=0
            for i in range(x.shape[1]):
                inner_sum += self.m_i[i]*x[n,i]
            outer_sum +=inner_sum+self.c-y[n]

        return (1/self.N)*outer_sum



    def cost_function_derivative_m(self, y, x, i):
        product1=LinearRegressor.cost_function_derivative_bias(self,y,x)
        product2 = np.sum(x[:,i])

        return (1/self.N)*product2*product1

    def update_weights(self, y, x):
        bias_derivative = LinearRegressor.cost_function_derivative_bias(self,y, x)
        new_c = self.c - (self.learning_rate) * bias_derivative
        self.c = new_c
        for i in range(len(self.m_i)):
            self.m_i[i] = self.m_i[i] - (self.learning_rate)*LinearRegressor.cost_function_derivative_m(self,y, x, i)

    def multi_dim_gradient_descent(self,y,x):
        active_tolerance=0
        rmse_0=LinearRegressor.multi_dim_error(self,y,x)
        print(f"{0}th iter rmse : {rmse_0} ")
        iter=0
        while active_tolerance<self.tolerance:
            LinearRegressor.update_weights(self,y,x)
            rmse_1=LinearRegressor.multi_dim_error(self,y,x)
            print(f"{iter+1}th iter rmse : {rmse_1} ")
            iter+=1

            if np.abs(rmse_1-rmse_0)<=self.error_threshold:
                active_tolerance+=1
            rmse_0 = rmse_1


































