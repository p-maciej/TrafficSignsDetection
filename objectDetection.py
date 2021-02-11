import numpy as np
import cv2 as cv

class objectDetection:

    def __init__(self):
        self.__lower_range = []
        self.__upper_range = []
        self.__font = cv.FONT_HERSHEY_SIMPLEX
        self.__boundingBoxPadding = 100
        self.__threshold = 100
        self.__fontSize = 1

    def videoCapture(self, model, windowHeight):
        cap = cv.VideoCapture(0)

        while True:
            _, img = cap.read()
            img = cv.resize(img, self.__scaleImage(img.shape[:2], windowHeight))
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            mask = cv.inRange(hsv, self.__lower_range[0], self.__upper_range[0])

            for i in range(1, len(self.__lower_range)):
                mask += cv.inRange(hsv, self.__lower_range[i], self.__upper_range[i])

            contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

            for contour in contours:
                approx = cv.approxPolyDP(contour, 0.1 * cv.arcLength(contour, True), True)
                (x, y, width, height) = cv.boundingRect(approx)


                if width > self.__threshold and height > self.__threshold and width/height > 0.9 and width/height < 1.1:
                    try:
                        cropped_image = img[y-self.__boundingBoxPadding:y+height+self.__boundingBoxPadding, x-self.__boundingBoxPadding:x+width+self.__boundingBoxPadding]
                        out = model.testImageFromArray(cropped_image)

                        if out != None:
                            cv.rectangle(img, (x-self.__boundingBoxPadding, y-self.__boundingBoxPadding), (x+width+self.__boundingBoxPadding, y+height+self.__boundingBoxPadding), (0, 255, 0), 2)
                            cv.putText(img, out, (x-self.__boundingBoxPadding, y-self.__boundingBoxPadding + 30), self.__font, 1, (0, 255, 0), 2)
                    except:
                        None
            cv.imshow('', img)
            k = cv.waitKey(5)
            if k == 27:
                break

        cap.release()
        cv.destroyAllWindows()

    def fromImage(self, model, pathImg, windowSize):
        img = cv.imread(pathImg)
        img = cv.resize(img, self.__scaleImage(img.shape[:2], windowSize))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv, self.__lower_range[0], self.__upper_range[0])

        for i in range(1, len(self.__lower_range)):
            mask += cv.inRange(hsv, self.__lower_range[i], self.__upper_range[i])

        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        for contour in contours:
            approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
            (x, y, width, height) = cv.boundingRect(approx)

            if width > self.__threshold and height > self.__threshold and width / height > 0.9 and width / height < 1.1:
                try:
                    cropped_image = img[y - self.__boundingBoxPadding:y + height + self.__boundingBoxPadding, x - self.__boundingBoxPadding:x + width + self.__boundingBoxPadding]
                    out = model.testImageFromArray(cropped_image)

                    if out != None:
                        cv.rectangle(img, (x - self.__boundingBoxPadding, y - self.__boundingBoxPadding), (
                        x + width + self.__boundingBoxPadding, y + height + self.__boundingBoxPadding), (0, 255, 0), 2)
                        cv.putText(img, out, (x - self.__boundingBoxPadding, y - self.__boundingBoxPadding + 30), self.__font, self.__fontSize, (0, 255, 0), 2)
                except:
                    None

        cv.imshow('', img)
        cv.waitKey()

    def setBoundingBoxPadding(self, size):
        self.__boundingBoxPadding = size

    def __getPoints(self, pictureSize, padding, points):
        x1 = points[0] - padding
        x2 = points[1] - padding
        x3 = points[2] + points[0] + padding
        x4 = points[3] + points[1] + padding

        if x1 < 0:
            x1 = 0
        if x2 < 0:
            x2 = 0
        if x3 > pictureSize[0] - 10:
            x3 = pictureSize[0] - 1
        if x4 > pictureSize[1] - 10:
            x4 = pictureSize[1] - 1

        return (x1, x2, x3, x4)

    def __scaleImage(self, size, desiredHeight):
        ratio = size[1] / size[0]
        return (int(desiredHeight * ratio), int(desiredHeight))

    def addLowerRange(self, range):
        self.__lower_range.append(range)

    def addUpperRange(self, range):
        self.__upper_range.append(range)

    def setThreshold(self, range):
        self.__threshold = range

    def setFontSize(self, size):
        self.__fontSize = size

    def setFont(self, font):
        self.__font = font