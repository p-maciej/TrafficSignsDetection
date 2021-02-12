import numpy as np

from model import trafficSigns
from objectDetection import objectDetection


model = trafficSigns()
model.loadNames('data/signnames.csv')
#model.loadModel('signsmodel')
model.loadData()
model.splitData(0.8)
model.train(12)
model.evaluate()
model.showTrainHistoryData("1.png", "2.png")
model.testImageFromPath('testdata/00007.png')

detection = objectDetection()

detection.addLowerRange(np.array([100, 50, 50])) # blue
detection.addUpperRange(np.array([130, 255, 255]))
detection.addLowerRange(np.array([0, 50, 50])) #red
detection.addUpperRange(np.array([10, 255, 255]))
detection.addLowerRange(np.array([170, 50, 50])) #red
detection.addUpperRange(np.array([180, 255, 255]))
detection.addLowerRange(np.array([26, 40, 30])) #yellow
detection.addUpperRange(np.array([34, 255, 255]))

# detection.setBoundingBoxPadding(30)
# detection.setThreshold(10)
# detection.setFontSize(1)
#detection.fromImage(model, "testdata/1.jpg", 800)
#detection.videoCapture(model, 800)