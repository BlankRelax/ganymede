# code for linear regressor
import  numpy as np
import matplotlib.pyplot as plt



class LinearRegressor:

    def __init__(self, learning_rate, error_threshold, tolerance):
        self.learning_rate=learning_rate
        self.error_threshold = error_threshold
        self.tolerance = tolerance
        self.m_i = []
        self.c = 1
        self.y_hat=[]


    def _error(self,y,x,m):
        squared_errors = []
        y_hat = []

        for i in range(len(y)):
            y_hat.append((m*x[i])+self.c)
            squared_errors.append(np.square((y[i]-y_hat[i])))

        rmse= np.sqrt(np.sum(squared_errors)/(2*len(y)))

        self.y_hat=y_hat
        return rmse

    def multi_dim_error(self,y,x):
        squared_errors = []
        y_hat = []

        for n in range(self.N):
            inner_sum=0
            for i in range(len(self.m_i)):
                inner_sum+=self.m_i[i]*x[n,i]+self.c
            y_hat.append(inner_sum)
            squared_errors.append(np.square((y[n]-y_hat[n])))

        rmse = np.sqrt(np.sum(squared_errors) / (2 * self.N))
        self.y_hat=y_hat
        return rmse




    def fit(self, y,x):

        #method current supports 1-D Linear regression fit

        self.N = len(y) # number of observations

        if type(y)!=np.ndarray:
            print('y should be a numpy.ndarray')
            exit()
        if type(x)!=np.ndarray:
            print('x should be a numpy.ndarray')
            exit()

        if y.shape[0]!=x.shape[0]:
            print("y and x are not of the same length")
        if len(x.shape)==1:
            self.m_i = [1]
            LinearRegressor.one_dim_gradient_descent(self,y,x)
        else:
            self.m_i = [1] * x.shape[1]
            LinearRegressor.multi_dim_gradient_descent(self,y,x)
            print(LinearRegressor.multi_dim_error(self,y,x))



    def _plot_line(self,y,x):
        plt.scatter(x,y,c='r', label='true data')
        plt.plot(x,self.y_hat, c='c',label='regression model')
        plt.title('Linear regression prediction against true data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='lower right')
        plt.show()
        plt.clf()

    def calculate_sum_row(self, x):
        return np.sum(x, 0)
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
        for i in range(1000):
            LinearRegressor.update_weights(self,y,x)
            print(f"{i} : {LinearRegressor.multi_dim_error(self,y,x)} ")

    def one_dim_gradient_descent(self,y,x):

        m=self.m_i[0]
        active_tolerance=0
        while active_tolerance<self.tolerance:
            rmse = LinearRegressor._error(self,y=y,x=x,m=m)
            rmse_plus = LinearRegressor._error(self,y=y,x=x,m=m+self.learning_rate)
            rmse_minus= LinearRegressor._error(self,y=y,x=x,m=m-self.learning_rate)

            rmse_spectrum = {'rmse':rmse, 'rmse_plus':rmse_plus, 'rmse_minus':rmse_minus}

            min_rmse=min(list(rmse_spectrum.values()))
            direction_arg = list(rmse_spectrum.values()).index(min_rmse)
            direction = list(rmse_spectrum.keys())[direction_arg]


            if direction=='rmse_plus':
                m = m+self.learning_rate
            elif direction=='rmse_minus':
                m=m-self.learning_rate

            self.m_i[0]=m

            print(f"rmse: {LinearRegressor._error(self, y=y, x=x, m=m)}")
            LinearRegressor._plot_line(self, y=y, x=x)

            if np.abs(min_rmse-rmse)<=self.error_threshold:
                active_tolerance+=1
            else:
                active_tolerance=0
































