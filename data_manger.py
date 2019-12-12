import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression



class data_mangment:
    def __init__(self):
        self.TRAIN_DIR = 'data\\train'
        self.TEST_DIR = 'data\\test'
        self.classes = {'airport_inside': 0, 'bakery': 1, 'bedroom': 2, 'greenhouse': 3, 'gym': 4, 'kitchen': 5,
                        'operating_room': 6, 'poolinside': 7, 'restaurant': 8, 'toystore': 9}
        self.IMG_SIZE = 224
        #self.LR = 0.001
        #self.MODEL_NAME = 'indoor_classifier'


    def create_label(self, index):
        """ Create an one-hot encoded vector from image name """
        lable=np.zeros(10)
        lable[index] = 1
        return lable


    def read_data(self):
        data = []
        for folder in tqdm(os.listdir(self.TRAIN_DIR)):
            folder_path = os.path.join(self.TRAIN_DIR, folder)
            for img in os.listdir(folder_path):
                image_path = os.path.join(folder_path, img)
                img_data = cv2.imread(image_path, 0)
                try:
                    img_data = cv2.resize(img_data, (self.IMG_SIZE, self.IMG_SIZE))
                    data.append([np.array(img_data), self.create_label(self.classes[folder])])
                except Exception as e:
                    print(img)

        training_data, testing_data = train_test_split(data, test_size=0.2, random_state=123, shuffle=True)
        np.save('train_data.npy', training_data)
        np.save('test_data.npy', testing_data)

        return training_data,testing_data

    def create_data(self):
        if (os.path.exists('train_data.npy')):  # If you have already created the dataset:
            np_load_old = np.load
            np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
            training_data = np.load('train_data.npy')
            testing_data = np.load('test_data.npy')
            np.load = np_load_old
        else:  # If dataset is not created:
            training_data,testing_data = self.read_data()
        return training_data, testing_data