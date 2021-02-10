import numpy as np
import cv2 as cv

class objectDetection:

    def __init__(self):
        self.lower_blue = np.array([0, 0, 0])
        self.upper_blue = np.array([0, 0, 0])
        self.lower_red = np.array([0, 0, 0])
        self.upper_red = np.array([0, 0, 0])
        self.lower_yellow = np.array([0, 0, 0])
        self.upper_yellow = np.array([0, 0, 0])
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.__boundingBoxPadding = 100

    def videoCapture(self, model, windowHeight):
        cap = cv.VideoCapture(0)

        while True:
            _, img = cap.read()
            img = cv.resize(img, self.__scaleImage(img.shape[:2], windowHeight))
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            mask_red = cv.inRange(hsv, self.lower_red, self.upper_red)
            mask_blue = cv.inRange(hsv, self.lower_blue, self.upper_blue)
            mask_yellow = cv.inRange(hsv, self.lower_yellow, self.upper_yellow)
            mask = mask_red + mask_blue + mask_yellow

            contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

            if len(contours) > 0:
                area = max(contours, key=cv.contourArea)
                (x, y, width, height) = cv.boundingRect(area)
                points = self.__getPoints(img.shape[:2], self.__boundingBoxPadding, (x, y, width, height))
                cropped_image = img[points[1]:points[3], points[0]:points[2]]
                out = model.testImageFromArray(cropped_image)
                if out != None:
                    cv.rectangle(img, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 2)
                    cv.putText(img, out, (points[0], points[1] + 30), self.font, 1, (0, 255, 0), 2)

            cv.imshow('', img)
            k = cv.waitKey(5)
            if k == 27:
                break

        cap.release()
        cv.destroyAllWindows()

    def fromImage(self, model, pathImg, windowSize, mode):
        img = cv.imread(pathImg)
        img = cv.resize(img, self.__scaleImage(img.shape[:2], windowSize))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        mask_red = cv.inRange(hsv, self.lower_red, self.upper_red)
        mask_blue = cv.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_yellow = cv.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = mask_red + mask_blue + mask_yellow

        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        if mode == False:
            for contour in contours:
                approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
                (x, y, width, height) = cv.boundingRect(approx)
                points = self.__getPoints(img.shape[:2], self.__boundingBoxPadding, (x, y, width, height))

                cropped_image = img[points[1]:points[3], points[0]:points[2]]
                out = model.testImageFromArray(cropped_image)

                cv.putText(img, out, (points[0], points[1] + 30), self.font, 1, (0, 255, 0), 2)
                cv.rectangle(img, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 3)
        else:
            area = max(contours, key=cv.contourArea)
            (x, y, width, height) = cv.boundingRect(area)
            points = self.__getPoints(img.shape[:2], self.__boundingBoxPadding, (x, y, width, height))

            cropped_image = img[points[1]:points[3], points[0]:points[2]]
            out = model.testImageFromArray(cropped_image)

            cv.putText(img, out, (points[0], points[1] + 30), self.font, 1, (0, 255, 0), 2)
            cv.rectangle(img, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 3)


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