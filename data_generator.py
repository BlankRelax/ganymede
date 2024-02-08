import  numpy as np

class data_generator:

    def generate_linear_data(self, start,stop,noise,m,c)->(np.ndarray,np.ndarray):

        # returns a tuple of numpy arrays (y,x) with dimensions
        x = np.linspace(start=start, stop=stop)
        delta= np.random.uniform(low=start/noise, high=stop/noise, size=len(x))
        y= (m*x)+c+delta

        return y,x







