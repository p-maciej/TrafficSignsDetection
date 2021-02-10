import cv2 as cv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


class trafficSigns:
    def __init__(self):
        self.model = models.Sequential()

    def __resizeImagesFromDir(self, dir, files, prefix = '', size=(32, 32)):
        x = np.array([cv.resize(cv.imread(prefix + fileName), size, interpolation=cv.INTER_AREA) for fileName in files])
        y = np.array([dir] * len(files))

        return (x, y)

    def loadData(self, size=(32, 32)):
        self.trainX, self.trainY = self.__resizeImagesFromDir(0, glob.glob('data/Train/'+str(0)+'/*.png'))

        for label in range(1, 43):
            print(label)
            filelist = glob.glob('data/Train/'+str(label)+'/*.png')
            x, y = self.__resizeImagesFromDir(label, filelist)
            self.trainX = np.concatenate((self.trainX ,x))
            self.trainY = np.concatenate((self.trainY ,y))

        ## normalization
        self.trainXnormalized = self.__normalizeData(self.trainX)

        files = pd.read_csv('data/Test.csv')['Path']
        self.testX,_ = self.__resizeImagesFromDir(0, files, 'data/')
        self.testY = np.array(pd.read_csv('data/Test.csv')['ClassId'])

        ## normalization
        self.testXnormalized = self.__normalizeData(self.testX)


    def splitData(self, rate):
        indices = np.random.permutation(self.trainX.shape[0])
        split_idx = int(self.trainX.shape[0] * rate)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        self.trainXReady, self.validationXReady = self.trainXnormalized[train_idx, :], self.trainXnormalized[val_idx, :]
        self.trainYReady, self.validationYReady = self.trainY[train_idx], self.trainY[val_idx]

    def loadNames(self, path):
        self.signNames = pd.read_csv(path)

    def __normalizeData(self, data): # to 0 - 255 to 0-1
        sum = np.sum(data/3, axis=3, keepdims=True)

        return (sum)/256

    def train(self, epochs):
        # Conv 32x32x1 => 28x28x6.
        self.model.add(layers.Conv2D(filters = 6, kernel_size = (5, 5), strides=(1, 1), padding='valid', activation='relu', data_format = 'channels_last', input_shape = (32, 32, 1)))
        # Maxpool 28x28x6 => 14x14x6
        self.model.add(layers.MaxPooling2D((2, 2)))
        # Conv 14x14x6 => 10x10x16
        self.model.add(layers.Conv2D(16, (5, 5), activation='relu'))
        # Maxpool 10x10x16 => 5x5x16
        self.model.add(layers.MaxPooling2D((2, 2)))
        # Flatten 5x5x16 => 400
        self.model.add(layers.Flatten())
        # Fully connected 400 => 120
        self.model.add(layers.Dense(120, activation='relu'))
        # Fully connected 120 => 84
        self.model.add(layers.Dense(84, activation='relu'))
        # Dropout
        self.model.add(layers.Dropout(0.2))
        # Fully connected, output layer 84 => 43
        self.model.add(layers.Dense(43, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # training batch_size=128, epochs=10
        self.model.fit(self.trainXReady, self.trainYReady, batch_size=128, epochs=epochs, validation_data=(self.validationXReady, self.validationYReady))
        #
        # acc = [conv.history['accuracy'], conv.history['val_accuracy']]
        # loss = [conv.history['loss'], conv.history['val_loss']]
        #
        # epoch = range(10)
        #
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.plot(epoch, acc[0], label='Training Accuracy')
        # plt.plot(epoch, acc[1], label='Validation Accuracy')
        # plt.legend(loc='lower right')
        # plt.title('Training and Validation Accuracy')
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(epoch, loss[0], label='Training Loss')
        # plt.plot(epoch, loss[1], label='Validation Loss')
        # plt.legend(loc='upper right')
        # plt.title('Training and Validation Loss')

    def evaluate(self):
        self.model.evaluate(x=self.testXnormalized, y=self.testY)

    def saveModel(self, modelname):
        self.model.save(modelname)

    def loadModel(self, modelname):
        self.model = tf.keras.models.load_model(modelname)

    def testImageFromPath(self, image):
        image = cv.resize(cv.imread(image), (32, 32), interpolation=cv.INTER_AREA)

        sum = np.sum(image / 3, axis=2, keepdims=True)
        image = (sum) / 256

        plt.title(self.signNames.loc[self.signNames['ClassId'] == np.argmax(self.model.predict(np.array([image]))), 'SignName'].values[0])
        plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

    def testImageFromArray(self, image):
        if (image.shape[:2][0] > 32 and image.shape[:2][1] > 32):
            image = cv.resize(image, (32, 32), interpolation=cv.INTER_AREA)

            sum = np.sum(image / 3, axis=2, keepdims=True)
            image = (sum - 128) / 128

            return self.signNames.loc[self.signNames['ClassId'] == np.argmax(self.model.predict(np.array([image]))), 'SignName'].values[0]
        return None