import  numpy as np

class data_generator:

    def generate_linear_data(self, start,stop,noise,m,c)->(np.ndarray,np.ndarray):

        # returns a tuple of numpy arrays (y,x) with dimensions
        x = np.linspace(start=start, stop=stop)
        delta= np.random.uniform(low=start/noise, high=stop/noise, size=len(x))
        y= (m*x)+c+delta

        return y,x

    def generate_multivariate_linear_data(self,number_of_variables,m:np.ndarray,c):

        if number_of_variables != len(m):
            print('the number of variables is not equal to the number of weights provided')
            exit()
        x = []
        y = []
        for i in range(number_of_variables):
            # generate x
            x.append(np.random.rand(200,1))
        x = np.array(x).reshape((200,len(m)))
        # generate y

        for i in range(200):
             y.append(np.dot(a=m, b=x[i,:].T)+c)

        return np.array(y), x














