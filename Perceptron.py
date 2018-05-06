# SUSHMITHA SHRIDHAR
# MACHINE LEARNING HW#1

import numpy as np
import pandas as pd
import gzip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
learning_rate = 0.1
#learning_rate = 0.01
#learning_rate = 0.001


########################################################## TRAIN ########################################################

f = gzip.open("t10k-images-idx3-ubyte.gz", 'rb')
test_images = np.frombuffer(f.read(), np.uint8, offset=16)
test_images = test_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(test_images.shape[0],784)

f = gzip.open("t10k-labels-idx1-ubyte.gz", 'rb')
test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
test_labels = test_labels.reshape(10000,1)

test_predicted = test_labels.transpose().astype(int).reshape(10000,)

#convert lables with index 1
test_labels_final = np.zeros([10000,10],dtype=int)

for row in range(len(test_images)):
    index = test_labels[row].astype(int)
    test_labels_final[row][index[0]] += 1

#preprocessing
test_images_matrix = np.copy(test_images)
test_images_matrix = test_images_matrix / np.float32(255)
bias_col = np.ones((10000,1))
test_images_matrix = np.append(test_images_matrix, bias_col,axis = 1)

test_y = np.zeros((10000,1))
test_actual = np.zeros((1,10000))

y_current = np.zeros((10000,10))
accuracy_test = np.zeros((70))

f = gzip.open("train-images-idx3-ubyte.gz", 'rb')
train_images = np.frombuffer(f.read(), np.uint8, offset=16)
train_images = train_images.reshape(-1, 1, 28, 28)
train_images = train_images.reshape(train_images.shape[0],784)

f = gzip.open("train-labels-idx1-ubyte.gz", 'rb')
train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
train_labels = train_labels.reshape(60000,1)


train_predicted = train_labels.transpose().astype(int).reshape(60000,)


#convert lables with index 1
train_labels_final = np.zeros([60000,10],dtype=int)

for row in range(len(train_images)):
    index = train_labels[row].astype(int)
    train_labels_final[row][index[0]] += 1

#preprocessing
train_images_matrix = np.copy(train_images)
train_images_matrix = train_images_matrix / np.float32(255)
bias_col = np.ones((60000,1))
train_images_matrix = np.append(train_images_matrix, bias_col,axis = 1)


# creating weights
weights = np.random.uniform(-0.05, 0.05, size=(10,785))

train_y = np.zeros((60000,1))
train_actual = np.zeros((1,60000))

y_current = np.zeros((60000,10))
accuracy_train = np.zeros((70))

for i in range(70):
    for row in range(len(train_images_matrix)):
        y_final = np.zeros([1,10],dtype=int)
        y = np.matmul(train_images_matrix[row:row+1],np.transpose(weights))
        y = np.matmul(train_images_matrix[row:row+1],np.transpose(weights))
        maxindex = np.argmax(y, axis = 1)
        y_final[0,maxindex[0]] = 1
        delta = train_labels_final[row:row+1] - y_final

        #print "train_labels_final = {}, train_images_matrix = {}".format(train_labels_final[row:row+1] ,(train_images_matrix[row:row+1]))

        index_y = y_final.argmax(axis = 1)
        train_y[row] = index_y


        if(delta.any(axis=1)):
        # print("update needed")
            weights = weights + 0.1 * np.matmul(np.transpose(delta),train_images_matrix[row:row+1])
        else:
            continue
         #print("no weight update needed")
            

#print y_final

    train_actual = train_y.transpose().reshape(60000,).astype(int)

    cfm_train = confusion_matrix(train_actual, train_predicted)

    #print cfm

    diagonal_sum_train =  sum(np.diag(cfm_train))
    #print diagonal_sum_train
    accuracy_train[i] = (diagonal_sum_train/60000.00)*100
    #print accuracy_train[i]

############################################################ Test ######################################################

    for row in range(len(test_images_matrix)):
        y_final = np.zeros([1,10],dtype=int)
        y = np.matmul(test_images_matrix[row:row+1],np.transpose(weights))
        maxindex = np.argmax(y, axis = 1)
        y_final[0,maxindex[0]] = 1
        delta = test_labels_final[row:row+1] - y_final

        index_y = y_final.argmax(axis = 1)
        test_y[row] = index_y

    test_actual = test_y.transpose().reshape(10000,).astype(int)

    cfm_test = confusion_matrix(test_actual, test_predicted)

    #print cfm

    diagonal_sum_test =  sum(np.diag(cfm_test))
    #print diagonal_sum_test
    accuracy_test[i] = (diagonal_sum_test/10000.00)*100
    #print accuracy_test[i]

    
########################################################### PLOTS ##########################################################
plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.ylabel("Accuracy in %")
plt.xlabel("Epoch")

image= "learningrate01.png"
plt.title("learning rate 0.1")
plt.savefig(image)
plt.show()

print "CONFUSION MATRIX OF TRAIN SET : learning rate 0.1 \n"
print cfm_train

print "CONFUSION MATRIX OF TEST SET : learning rate 0.1 \n"
print cfm_test

