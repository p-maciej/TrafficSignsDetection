import numpy as np

from model import trafficSigns
from objectDetection import objectDetection


model = trafficSigns()

### Loading names ####
model.loadNames('data/signnames.csv')


## Loading trained model ####
model.loadModel('signsmodel')

###### MODEL TRAINING ###########
# model.loadData()
# model.splitData(0.8)
# model.train(12)
# model.evaluate()
# model.showTrainHistoryData("1.png", "2.png")
# model.testImageFromPath('testdata/00007.png')
# model.saveModel("signsmodel")
##################################


detection = objectDetection()


### Detection of colors ###
detection.addLowerRange(np.array([95, 15, 15])) # blue
detection.addUpperRange(np.array([130, 255, 255]))
detection.addLowerRange(np.array([0, 35, 35])) #red
detection.addUpperRange(np.array([10, 255, 255]))
detection.addLowerRange(np.array([170, 35, 35])) #red
detection.addUpperRange(np.array([180, 255, 255]))
detection.addLowerRange(np.array([20, 50, 50])) #yellow
detection.addUpperRange(np.array([30, 255, 255]))

detection.setBoundingBoxPadding(100)
detection.setThreshold(100)
detection.setFontSize(1)


### Sings detection based on picture ###
# detection.fromImage(model, "testdata/1.jpg", 800)

### Sings detection from video capture ###
detection.videoCapture(model, 900)