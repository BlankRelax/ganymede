import  numpy as np

class data_generator:

    def generate_linear_data(self, start,stop,m,c)->(np.ndarray,np.ndarray):

        # returns a tuple of numpy arrays (y,x) with dimensions
        x = np.linspace(start=start, stop=stop)
        delta= np.random.uniform(low=0, high=50, size=len(x))
        y= (m*x)+c+delta

        return y,x







