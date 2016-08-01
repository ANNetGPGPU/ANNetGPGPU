from ANPyNetCPU import *
black 	= vectorf([0,0,0])
white 	= vectorf([1,1,1])
red 	= vectorf([1,0,0])
green 	= vectorf([0,1,0])
blue 	= vectorf([0,0,1])

trainSet = TrainingSetF()
trainSet.AddInput(black)
trainSet.AddInput(white)
trainSet.AddInput(red)
trainSet.AddInput(green)
trainSet.AddInput(blue)

widthMap = 5
heightMap = 1

inpWidth = 3
inpHeight = 1

SOM = SOMNetGaussF(inpWidth,inpHeight,widthMap,heightMap)
SOM.SetTrainingSet(trainSet)
SOM.SetLearningRate(0.75)
SOM.Training(100)

# gets an ordered list of different centroids with the ID of the corresponding BMU
centroids = SOM.GetCentroidList()

# output for fun
for i in centroids:
  print (i)
