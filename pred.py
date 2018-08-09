# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:19:06 2018

@author: User
"""
import time
import cv2
from scipy.io import loadmat
import pickle
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy
from keras.models import model_from_json
K.set_image_dim_ordering('th')
from sklearn.model_selection import train_test_split


#def modify(img, threshold=0.5):
#    for row in range(28):
#        for col in range(28):
#            if ((img[0][row][col] < threshold) and (row>3 and row < 24) and (col>3 and col < 24)):
#                img[0][row][col] = 1
#            else:
#                img[0][row][col] = 0

def modify(img, threshold=0.5):
    for row in range(28):
        for col in range(28):
#            if((row < 4) or (row>24) or (col < 4) or (col > 24)):
#                img[0][row][col] = 0
            if ((img[0][row][col] >= threshold)):
                img[0][row][col] = 1
            else:
                img[0][row][col] = 0

def display(img, threshold=0.5):
    # Debugging only
#    render = ''
#    for row in range(28):
#        for col in range(28):
#            if img[0][row][col] > threshold:
#                render += '@'
#            else:
#                render += '.'
#        render += '\n'
#    print(render)
    
    
    image = cv2.imread('test.bmp')
    image = cv2.resize(image, (28, 28))
    
    for row in range(28):
        for col in range(28):
            if img[0][row][col] > threshold:
                image[row][col]=1
            else:
                image[row][col]=0
    
   
   
      
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    plt.imshow(image, cmap='gray', interpolation='bicubic')
    plt.show()
    
#
#dataset = numpy.loadtxt("DigitTrain_1.txt", delimiter=",")
#print(dataset.shape)
#dim = 28*28
#X = dataset[:,0:dim]
#y = dataset[:,dim]

dim = 28*28
dataset = numpy.loadtxt("test1.txt" , delimiter = ",")
X = dataset[:,0:dim]
Y = dataset[:,dim]
seed = 785
(X_train_our, X_test_our, y_train_our, y_test_our) = train_test_split(X, Y, test_size=0.20, random_state=seed)

X = X.reshape(X.shape[0] , 1 , 28 , 28).astype('float32')

for i  in range(X.shape[0]):
    modify(X[i])

for i  in range(min(3000 , X.shape[0])):
#    print(" ====> " , int(y[i]))
#    print(i,X[i].shape,int(y[i]))
    display(X[i])


json_file = open('model_bangla_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_bangla_character_recognition.h5")
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

dim = 28*28
test_data = numpy.loadtxt("Test.txt", delimiter=",")
X_test = test_data[:,0:dim]
y_test = test_data[:,dim]
X_test = X_test.reshape(X_test.shape[0] , 1 , 28 , 28).astype('float32')
yy = y_test
y_test = np_utils.to_categorical(y_test)


#X_test = X
#yy = y
# =============================================================================
# image = cv2.imread('test.bmp')
# image = cv2.resize(image, (28, 28))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# =============================================================================


num_classes = 50
match = [0]*num_classes
not_match = [0]*num_classes

c1 = 0

prediction = model.predict(X_test)
for i in range(X_test.shape[0]):
    now = -1
    mx = -1
    mx2 = -1
    now2 = -1
    for j in range(num_classes):
        if(prediction[i][j] > mx): 
            now = j;
            mx = prediction[i][j]
        elif(prediction[i][j] > mx2): 
            now2 = j;
            mx2 = prediction[i][j]
#    display(X_test[i])
    ch1 = now
    ch2 = now2
    ans = yy[i];
#    print( " ------------------> " , mx , ch1 ,"  2nd choice:", mx2 , ch2)
    idx = int(yy[i])
    if(ch1 == ans):
        match[idx]+=1
    else:
        not_match[idx]+=1
        display(X_test[i])
        print("Actual:",ans,"1st prediction:",mx,ch1,"2nd Pred:",mx2,ch2)
        time.sleep(1)

for i in range(num_classes):
    print(i , " ===> " , match[i] , not_match[i])

accuracy = (sum(match)/(sum(match) + sum(not_match)))*100.0
print("accuracy = " , accuracy , '%')   

