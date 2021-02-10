import numpy as np
import cv2 as cv
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from modelTrain import trafficSigns

def scaleImage(size, desiredHeight):
    ratio = size[1]/size[0]
    return (int(desiredHeight*ratio), int(desiredHeight))

def addThreshoolToMask(mask, size):
    print()

def resizeImagesFromDir(dir, files, size=(32, 32)):
    x = np.array([cv.resize(cv.imread(fileName), size, interpolation=cv.INTER_AREA) for fileName in files])
    y = np.array([dir] * len(files))

    return (x, y)

def loadNames():
    return pd.read_csv('data/signnames.csv')

def normalizeData(data):
    sum = np.sum(data/3, axis=3, keepdims=True)

    return (sum-128)/128

def loadmodel(modelname):
    return tf.keras.models.load_model(modelname)

def testImage(model, signNames, image):
    image = cv.resize(cv.imread(image), (32, 32), interpolation=cv.INTER_AREA)

    sum = np.sum(image / 3, axis=2, keepdims=True)
    image = (sum - 128) / 128

    #print(np.argmax(model.predict(np.array([image]))))
    plt.title(signNames.loc[signNames['ClassId'] == np.argmax(model.predict(np.array([image]))), 'SignName'].values[0])
    plt.imshow(image.squeeze(), cmap='gray')
    plt.show()

def testImageArr(model, signNames, image):
    if(image.shape[:2][0] > 32 and image.shape[:2][1] > 32):
        image = cv.resize(image, (32, 32), interpolation=cv.INTER_AREA)

        sum = np.sum(image / 3, axis=2, keepdims=True)
        image = (sum - 128) / 128

        return signNames.loc[signNames['ClassId'] == np.argmax(model.predict(np.array([image]))), 'SignName'].values[0]
    return None

def getPoints(pictureSize, padding, points):
    x1 = points[0]-padding
    x2 = points[1]-padding
    x3 = points[2]+points[0]+padding
    x4 = points[3]+points[1]+padding

    if x1 < 0:
        x1 = 0
    if x2 < 0:
        x2 = 0
    if x3 > pictureSize[0]-10:
        x3 = pictureSize[0]-1
    if x4 > pictureSize[1]-10:
        x4 = pictureSize[1]-1

    return (x1, x2, x3, x4)

def videoCapture():
    cap = cv.VideoCapture(0)
    signsNames = loadNames()
    model = loadmodel("signsmodel")
    font = cv.FONT_HERSHEY_SIMPLEX
    padding = 100
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([25, 50, 50])
    upper_yellow = np.array([55, 255, 255])

    while True:
        _, img = cap.read()
        img = cv.resize(img, scaleImage(img.shape[:2], 800))
        #img = cv.blur(img, (6, 6))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        mask_red = cv.inRange (hsv, lower_red, upper_red)
        mask_blue = cv.inRange (hsv, lower_blue, upper_blue)
        mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
        mask = mask_red + mask_blue + mask_yellow

        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) > 0:
            area = max(contours, key=cv.contourArea)
            (x,y,width,height) = cv.boundingRect(area)
            points = getPoints(img.shape[:2], padding, (x, y, width, height))
            cropped_image = img[points[1]:points[3], points[0]:points[2]]
            out = testImageArr(model, signsNames, cropped_image)
            if out != None:
                cv.rectangle(img, (points[0],points[1]),(points[2], points[3]), (0, 255, 0), 2)
                cv.putText(img, out, (points[0],points[1]+30), font, 1, (0, 255, 0), 2)

        cv.imshow('', img)
        k = cv.waitKey(5)
        if k == 27:
            break


    cap.release()
    cv.destroyAllWindows()

def main():
    img = cv.imread("1.jpg")
    img = cv.resize(img, scaleImage(img.shape[:2], 800))
    img = cv.blur(img, (6, 6))
    #cap = cv.VideoCapture(0)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([130,255,255])
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    lower_yellow = np.array([25, 93, 0])
    upper_yellow = np.array([45, 255, 255])

    font = cv.FONT_HERSHEY_SIMPLEX




    mask_red = cv.inRange (hsv, lower_red, upper_red)
    mask_blue = cv.inRange (hsv, lower_blue, upper_blue)
    mask_yellow = cv.inRange (hsv, lower_yellow, upper_yellow)
    mask = mask_red + mask_blue

    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    padding = 50

    signsNames = loadNames()


    model = loadmodel("signsmodel")

    #testImage(model, signsNames, "00007.png")

    #method1
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
        (x,y,width,height) = cv.boundingRect(approx)
        points = getPoints(img.shape[:2], padding, (x, y, width, height))


        cropped_image = img[points[1]:points[3], points[0]:points[2]]
        out = testImageArr(model, signsNames, cropped_image)

        cv.putText(img, out, (points[0], points[1] + 30), font, 1, (0, 255, 0), 2)
        cv.rectangle(img, (points[0],points[1]),(points[2], points[3]),(0,255,0),3)
        cv.imshow('', img)


        #method2
        # area = max(contours, key=cv.contourArea)
        # (x,y,width,height) = cv.boundingRect(area)
        # points = getPoints(img.shape[:2], padding, (x, y, width, height))
        # cropped_image = img[points[1]:points[3], points[0]:points[2]]
        # out = testImageArr(model, signsNames, cropped_image)
        # cv.rectangle(img, (points[0],points[1]),(points[2], points[3]), (0, 255, 0), 2)
        # cv.putText(img, out, (points[0],points[1]+30), font, 1, (0, 255, 0), 2)
        #
        # cv.imshow('', img)
    cv.waitKey()

def model():
    model = trafficSigns()

    model.loadNames('data/signnames.csv')
    model.loadModel('signsmodel')
    #model.loadData()
    #model.splitData(0.8)
    #model.train(10)
    model.testImage("00007.png")

model()