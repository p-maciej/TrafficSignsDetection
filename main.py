import numpy as np

from model import trafficSigns
from objectDetection import objectDetection


model = trafficSigns()
model.loadNames('data/signnames.csv')
#model.loadModel('signsmodel')
model.loadData()
model.splitData(0.8)
model.train(2)
model.evaluate()
model.testImageFromPath('testdata/00007.png')

detection = objectDetection()

detection.lower_blue = np.array([100, 50, 50])
detection.upper_blue = np.array([130, 255, 255])
detection.lower_red = np.array([0, 50, 50])
detection.upper_red = np.array([10, 255, 255])
detection.lower_yellow = np.array([25, 50, 50])
detection.upper_yellow = np.array([55, 255, 255])
detection.setBoundingBoxPadding(100)

#detection.fromImage(model, "testdata/3.jpg", 800, True)
#detection.videoCapture(model, 800)