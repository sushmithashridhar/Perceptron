import numpy as np
import pandas as pd
import gzip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt




f = gzip.open("train-images-idx3-ubyte.gz", 'rb')
train_images = np.frombuffer(f.read(), np.uint8, offset=16)
train_images = train_images.reshape(-1, 1, 28, 28)
train_images = train_images.reshape(train_images.shape[0],784)

bias_col = np.ones((60000,1))


f = gzip.open("train-labels-idx1-ubyte.gz", 'rb')
train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
train_labels = train_labels.reshape(60000,1)

#convert lables with index 1
train_labels_final = np.zeros((60000,10))

for row in range(len(train_images)):
	index = train_labels[row].astype(int)
	train_labels_final[row][index-1] += 1


# creating a single row of target values in a confusion matrix
train_predicted = train_labels.transpose().astype(int).reshape(60000,)
print train_predicted.shape
print train_predicted

#preprocessing
train_images_matrix = np.copy(train_images)
train_images_matrix[:,:] /= 255
# bias_col = np.ones((60000,1))
train_images_matrix = np.append(train_images_matrix, bias_col,axis = 1)


# train_images_matrix.astype(int)

# creating weights
weights = np.random.uniform(-0.05, 0.05, size=(10,785))
#print weights.round(2)

train_y = np.zeros((60000,1))
train_actual = np.zeros((1,60000))


y_current = np.zeros((60000,10))
for i in range(5):
	for row in range(len(train_images_matrix)):
		for perceptron in range(10):
			y = np.dot(train_images_matrix[row,:],weights[perceptron].T)
			y_current[row][perceptron] += y
		
y_final = (y_current == y_current.max(axis = 1, keepdims = 1)).astype(int)

y_final_index = np.zeros((60000,1))

for row in range(len(y_final)):
	y_final_index[row] = y_final[row].argmax(axis = 0)

counter = 0
accuracy = 0


print y_final_index.shape
print y_final_index
for row in range(len(train_labels)):
	#print "actual label = {}, computed label = {}".format(train_labels[row],y_final_index[row].astype(int))
	a = y_final_index.reshape(60000,1).astype(int)
	if (train_labels[row] == y_final_index[row].astype(int)):
		accuracy += 1
	else:
		counter += 1	
print accuracy
print counter


	#comparing two matrix

	#if not np.array_equal(train_labels_final, y_final.T):
	#	print "weight update"
	#	error = 0.0
	#	for perceptron in range(785):
	#		error = train_labels_final[perceptron] - y_final[perceptron]
	#		weights[perceptron] += 0.001 * np.dot(train_images_matrix[perceptron], weights[perceptron].reshape(1,785) * error)


index_y = y_final.argmax(axis = 1)


'''
			y_final = (y == y.max(axis = 1, keepdims = 1)).astype(int)


			index_y = y_final.argmax(axis = 1)
			train_y[row] = index_y

			#print "sub_matrix"
			sub_matrix = train_labels_final[row] - y_final 

			# print "actual label = {}, computed label = {}".format((train_labels_final[row] == train_labels_final[row].max(axis = 0, keepdims = 1)).astype(int),(y_final == y_final.max(axis = 1, keepdims = 1)).astype(int))
			# print train_labels_final[row].reshape(1,10).shape, y_final.shape
			if not np.array_equal(train_labels_final[row], y_final.T):
				# print "weight update...."
				#weights[j] = weights.transpose()
				#weights += 0.001 * sub_matrix.reshape(10,1).astype(int) * train_images_matrix[row]	
				#weights = weights.transpose()

		#print weights[row]

		#print train_images_matrix[row]


		

#print "New weights"
	#print weights


#generating confusion matrix from actual and predicted values
cfm = confusion_matrix(train_actual, train_predicted)
	#print cfm

	#calculating accuracy
	# diagonal_sum =  sum(np.diag(cfm))
	# print diagonal_sum
	# accuracy = (diagonal_sum/60000.00)*100
	# print accuracy

train_actual = train_y.transpose().reshape(60000,).astype(int)
print train_actual.shape
print train_actual
#print weights.round(2)
plt.plot(accuracy)
plt.show()



test_labels_final = np.zeros((10000,10))


for row in range(len(test_labels_final)):
	index = test_labels[row].astype(int)
	test_labels_final[index] += 1


print test_labels
print test_labels_final
'''
