import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
 
file1 = r'mnist_train.csv'
file2 = r'mnist_test.csv'

#df = pd.read_csv(file)
#print(pd.DataFrame(df))


inputdata = np.array(pd.read_csv(file1), dtype = float)

testdata = np.array(pd.read_csv(file2), dtype = float)

#print(pd.DataFrame(result))

inputdata[:,1:] /= 255;
testdata[:,1:] /= 255;

#print result[:,0]

Target_input = inputdata[:, :1]
Target_test = testdata[:, :1]

print "Target_input split"
print Target_input.shape

print "Target_test split"
print Target_test.shape


inputdata = inputdata[:, 1:]
testdata = testdata[:, 1:]
 
print "inputdata split"
print inputdata.shape

print "testdata split"
print testdata.shape

one = np.ones((59999,1))

one1 = np.ones((9999,1))

inputvalue = np.hstack((one,inputdata))
testvalue = np.hstack((one1,testdata))

print "Inputdata with 1"
print inputvalue.shape

print "Testdata with 1"
print testvalue.shape

#print(pd.DataFrame(result))
#print result
#weights = np.array(np.random.uniform(-0.05, 0.05, result.shape[10][1])

weights = np.random.uniform(low=-0.05, high=0.05, size=(785,10) )


print "Weights" 
print weights.shape


y = np.dot(inputvalue,weights)

print "Dot product of input and weight"
print y.shape


def Maxofdotproduct(inputvalue):
	y = np.dot(inputvalue[0],weights)
	print "y"
	print y.shape
	return np.argmax(y)

def accuracy(inputvalue):

	trainingConfusionMatrix = np.zeros((10,10))
	testConfusionMatrix = np.zeros((10,10))

	for i in range(0,1):
		index = Maxofdotproduct(inputvalue)
		print "For Train"
		print index
		trainingConfusionMatrix[inputvalue[i][0]][index] += 1


	for i in range(0,1):
		index = Maxofdotproduct(testvalue)
		print "For Test"
		print index
		testingConfusionMatrix[testvalue[i][0]][index] += 1


accuracy(inputvalue)
#print y

#a = np.sum(y)

#print a

#result = 1*np.andom.uniform(-0.05, 0.05)

#weights_transpose = weights.transpose()

#y = np.dot(result[:,0],weights)

#a = result[0,:]

#print a.shape
#
#print y
#
#print y.shape

#print weights
#print value

#print result[0]
#print value.shape
