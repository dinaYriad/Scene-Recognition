import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from data_manger import data_mangment

IMG_SIZE = 224
MODEL_NAME = 'indoor_classifier'
LR = 0.001

data=data_mangment()
training_data,testing_data = data.create_data()


X_train = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train = [i[1] for i in training_data]

X_test = np.array([i[0] for i in testing_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in testing_data]

tf.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

#Student code
conv6 = conv_2d(pool5, 64, 5, activation='relu')
pool6 = max_pool_2d(conv6, 5)
#Student code

fully_layer1 = fully_connected(pool6, 1024, activation='relu')
fully_layer1 = dropout(fully_layer1, 0.5)

fully_layer2 = fully_connected(fully_layer1, 1024, activation='relu')
fully_layer2 = dropout(fully_layer2, 0.5)

'''
fully_layer3 = fully_connected(fully_layer2, 1024, activation='relu')
fully_layer3 = dropout(fully_layer3, 0.5)

fully_layer4 = fully_connected(fully_layer3, 1024, activation='relu')
fully_layer4 = dropout(fully_layer4, 0.5)

fully_layer5 = fully_connected(fully_layer4, 1024, activation='relu')
fully_layer5 = dropout(fully_layer5, 0.5)
'''
cnn_layers = fully_connected(fully_layer2, 10, activation='softmax')
cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)


if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')

else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')

#Student code :Classify the images in the test_data and compute the overall accuracy
predictions_list=[]


c=0
for i in range(len(X_test)):
    prediction = model.predict([X_test[i]])[0]
    if prediction[0]>prediction[1]:
        predictions_list.append([1, 0])
    else:
        predictions_list.append([0, 1])
    if predictions_list[i][0]==y_test[i][0]:
        c+=1

test_accuracy = c/len(X_test)
print ("Test accuracy is " + str(test_accuracy*100))
#Student code

