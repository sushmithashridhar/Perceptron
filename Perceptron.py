import numpy as np
import pandas as pd
import gzip
from sklearn.metrics import confusion_matrix


f = gzip.open("train-images-idx3-ubyte.gz", 'rb')
train_images = np.frombuffer(f.read(), np.uint8, offset=16)
train_images = train_images.reshape(-1, 1, 28, 28)
train_images = train_images.reshape(train_images.shape[0],784)

bias_col = np.ones((60000,1))



f = gzip.open("train-labels-idx1-ubyte.gz", 'rb')
train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
train_labels = train_labels.reshape(60000,1)

print train_labels.shape

train_labels_final = np.zeros((60000,10))


#convert lables with index 1
for row in range(len(train_images)):
	index = train_labels[row].astype(int)
	train_labels_final[row][index-1] += 1

train_labels_final.astype(int)

print train_labels_final.shape

train_images_matrix = np.copy(train_images)
train_images_matrix[:,:] /= 255
bias_col = np.ones((60000,1))
train_images_matrix = np.append(train_images_matrix, bias_col,axis = 1)


train_images_matrix.astype(int)

#print train_images_matrix


confusion_matrix = np.zeros((10,10))

weights = np.random.uniform(-0.05, 0.05, size=(785,10))

#print weights.round(2)

z_final = np.zeros((60000,10))

for row in range(len(train_images_matrix)):
	y = np.dot(train_images_matrix[row,:],weights).reshape(1,10)
	#print y
	y_final = (y == y.max(axis = 1, keepdims = 1)).astype(int)

	z_final = np.append(z_final , y_final, axis = 0)

	print z_final

	#b = confusion_matrix(train_labels_final[row,:],y_final)
	#diagonal_sum =  sum(np.diag(b))
	#accuracy = (diagonal_sum/60000)*100
	#print accuracy

	sub_matrix = train_labels_final[row] - y_final 

	#print sub_matrix.astype(int)

	if (sub_matrix.any(axis=1)):
		weights = weights.transpose()
		weights += 0.001 * sub_matrix.reshape(10,1) * train_images_matrix[row]
		#print "weights changing" 	
		weights = weights.transpose()


#print "weights changed" 
#print weights.round(2)
#b = confusion_matrix(train_labels_final,y_final)
print z_final.shape
#print confusion_matrix

#diagonal_sum =  sum(np.diag(b))
#
#accuracy = (diagonal_sum/60000)*100
#print accuracy
#print column_id




'''
test_labels_final = np.zeros((10000,10))


for row in range(len(test_labels_final)):
	index = test_labels[row].astype(int)
	test_labels_final[index] += 1


print test_labels
print test_labels_final
'''
