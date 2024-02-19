import numpy as np
rand=np.random.randint(low=0, high=2,size=2)
x=[[3,4,5],[5,6,7],[5,7,9]]
x= np.array(x)
print(x[rand,:])