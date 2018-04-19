from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
 
file = r'mnist_train.csv'
#df = pd.read_csv(file)
#print(pd.DataFrame(df))


result = np.array(pd.read_csv(file), dtype = float)
#print(pd.DataFrame(result))

result[:,1:] /= 255;

print(pd.DataFrame(result))