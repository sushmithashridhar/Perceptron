import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
 
file = r'mnist_train.csv'
#df = pd.read_csv(file)
#print(pd.DataFrame(df))


result = np.array(pd.read_csv(file), dtype = float)
#print(pd.DataFrame(result))

result[:,1:] /= 255;

#print result[:,0]

target = result[:,0]

#print(pd.DataFrame(result))

#print result

#print(result.shape)

#weights = np.array(np.random.uniform(-0.05, 0.05, result.shape[10][1])

weights = np.random.uniform(low=-0.05, high=0.05, size=(784,10) )

#result = 1*np.andom.uniform(-0.05, 0.05)


y = np.dot(result[0,:],weights)

#print weights
#print value

#print result[0]
#print value.shape
