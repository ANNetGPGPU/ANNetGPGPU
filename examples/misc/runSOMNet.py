from ANPyNetCPU import *
black 	= vectorf([0,0,0])
white 	= vectorf([1,1,1])
red 	= vectorf([1,0,0])
green 	= vectorf([0,1,0])
blue 	= vectorf([0,0,1])

trainSet = TrainingSet()
trainSet.AddInput(black)
trainSet.AddInput(white)
trainSet.AddInput(red)
trainSet.AddInput(green)
trainSet.AddInput(blue)

widthMap = 4
heightMap = 1

inpWidth = 3
inpHeight = 1

SOM = SOMNet(inpWidth,inpHeight,widthMap,heightMap)
SOM.SetTrainingSet(trainSet)
SOM.SetLearningRate(0.3)
SOM.Training(1000)

# gets to each input vector the corresponding centroid, eucl. distance and the ID of the BMU
inputv = SOM.GetCentrOInpList()
# gets an ordered list of different centroids with the ID of the corresponding BMU
centroids = SOM.GetCentroidList()

# output for fun
for i in centroids:
  print (i)

# .. again
for i in inputv:
  print (i)
  
# Save IDs of the BMUs into a list
IDList = []
for i in inputv:
  IDList.append(i.m_iBMUID)
print (IDList)

# Searches the corresponding centroids from the other list based on the index :D
for i in IDList:
  for j in centroids:
    if i == j.m_iBMUID:
      print (j.m_vCentroid)