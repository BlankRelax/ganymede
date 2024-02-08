# code for linear regressor
import  numpy as np
import matplotlib.pyplot as plt


class LinearRegressor:

    def __init__(self, learning_rate, error_threshold):
        self.learning_rate=learning_rate
        self.error_threshold = error_threshold
        self.m = 3
        self.c = 0


    def _error(self,y,x,m,c):
        squared_errors = []
        y_hat = []
        for i in range(len(y)):
            y_hat.append(m*x[i]+c)
            squared_errors.append(np.square((y[i]-y_hat[i])))

        rmse= np.sqrt(np.sum(squared_errors)/len(y))

        self.y_hat=y_hat
        return rmse

    def fit(self, y,x):

        #method current supports 1-D Linear regression fit

        if type(y)!=np.ndarray:
            print('y should be a numpy.ndarray')
            exit()
        if type(x)!=np.ndarray:
            print('x should be a numpy.ndarray')
            exit()

        if y.shape[0]!=x.shape[0]:
            print("y and x are not of the same length")

        LinearRegressor.one_dim_gradient_descent(self,y,x)

    def _plot_line(self,y,x):
        plt.scatter(x,y,c='r', label='true data')
        plt.plot(x,self.y_hat, c='c',label='regression model')
        plt.title('Linear regression prediction against true data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='lower right')
        plt.show()

    def one_dim_gradient_descent(self,y,x):
        rmse = LinearRegressor._error(self, y=y, x=x, m=self.m, c=self.c)
        print(f"rmse: {rmse}")
        #TODO change while loop statement so it terminates if there is not a signficant change between successive rmse

        while np.abs(rmse)>self.error_threshold:
            rmse = LinearRegressor._error(self,y=y,x=x,m=self.m,c=self.c)
            rmse_plus = LinearRegressor._error(self,y=y,x=x,m=self.m+self.learning_rate,c=self.c)
            rmse_minus= LinearRegressor._error(self,y=y,x=x,m=self.m-self.learning_rate,c=self.c)

            rmse_spectrum = {'rmse':rmse, 'rmse_plus':rmse_plus, 'rmse_minus':rmse_minus}

            min_rmse=min(list(rmse_spectrum.values()))
            direction_arg = list(rmse_spectrum.values()).index(min_rmse)
            direction = list(rmse_spectrum.keys())[direction_arg]


            if direction=='rmse_plus':
                self.m = self.m+self.learning_rate
            elif direction=='rmse_minus':
                self.m=self.m-self.learning_rate

            print(f"rmse: {LinearRegressor._error(self, y=y, x=x, m=self.m, c=self.c)}")
            LinearRegressor._plot_line(self, y=y, x=x)
























