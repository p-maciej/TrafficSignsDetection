import cv2 as cv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


class trafficSigns:
    def __init__(self):
        self.model = models.Sequential()

    def __resizeImagesFromDir(self, files, prefix = '', size=(32, 32)):
        return np.array([cv.resize(cv.imread(prefix + fileName), size, interpolation=cv.INTER_AREA) for fileName in files])

    def loadData(self):
        print("Loading directory: 0")
        files = glob.glob('data/Train/0/*.png')
        self.trainX = self.__resizeImagesFromDir(files)
        self.trainY = np.array([0] * len(files))

        for label in range(1, 43):
            print("Loading directory: " + str(label))
            files = glob.glob('data/Train/' + str(label) + '/*.png')
            x = self.__resizeImagesFromDir(files)
            y = np.array([label] * len(files))

            self.trainX = np.concatenate((self.trainX ,x))
            self.trainY = np.concatenate((self.trainY ,y))

        ## normalization
        self.trainXnormalized = self.__normalizeData(self.trainX)

        print("Loading test data...")
        files = pd.read_csv('data/Test.csv')['Path']
        self.testX = self.__resizeImagesFromDir(files, 'data/')
        self.testY = np.array(pd.read_csv('data/Test.csv')['ClassId'])

        ## normalization
        self.testXnormalized = self.__normalizeData(self.testX)


    def splitData(self, rate):
        self.trainXReady, self.validationXReady,self.trainYReady, self.validationYReady = train_test_split(self.trainXnormalized, self.trainY, test_size=1-rate, shuffle=True)

    def loadNames(self, path):
        self.signNames = pd.read_csv(path)

    def __normalizeData(self, data): # to 0 - 255 to 0-1
        sum = np.sum(data/3, axis=3, keepdims=True)

        return (sum)/256

    def train(self, epochs):
        self.model.add(layers.Conv2D(filters = 8, kernel_size = (5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape = (32, 32, 1))) #tanh works well too
        self.model.add(layers.AvgPool2D((2, 2)))
        self.model.add(layers.Conv2D(32, (5, 5), activation='tanh'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(120, activation='tanh'))
        self.model.add(layers.Dense(84, activation='tanh'))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(43, activation='sigmoid'))

        self.model.summary()

        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        self.__model_fit = self.model.fit(self.trainXReady, self.trainYReady, batch_size=128, epochs=epochs, validation_data=(self.validationXReady, self.validationYReady))
        self.__epochs = epochs

    def showTrainHistoryData(self, pltName1, pltName2):
        accuracy = [self.__model_fit.history['accuracy'], self.__model_fit.history['val_accuracy']]
        loss = [self.__model_fit.history['loss'], self.__model_fit.history['val_loss']]

        plt.plot(range(self.__epochs), accuracy[0], label='Training Accuracy')
        plt.plot(range(self.__epochs), accuracy[1], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.savefig(pltName1)
        plt.close()

        plt.plot(range(self.__epochs), loss[0], label='Training Loss')
        plt.plot(range(self.__epochs), loss[1], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(pltName2)
        plt.close()

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