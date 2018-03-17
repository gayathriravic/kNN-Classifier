import csv
import random
import math
import operator

def loadCsvFile():
    trainingSet=[]
    testSet=[]
    with open('lenses.csv') as csvFile:
        lines=csv.reader(csvFile)
        dataset=list(lines)
        for i in range(0,len(dataset)-1):
            for j in range(0,len(dataset[i])-1):
                if random.random()< 0.8:       #split data : 80% for training and 20% for testing
                    trainingSet.append(dataset[i])
                else:
                    testSet.append(dataset[i])
    return testSet,trainingSet


def computeEuclideanDistance(trainingSet,test,length):
    dist=0
    for i in range(length-1):
        dist+=pow((float(trainingSet[i])-float(test[i])),2)
    return math.sqrt(dist)


def getSimilarity(trainingSet, test,k):
    distanceList=[]
    for i in range(len(trainingSet)):
      distance=computeEuclideanDistance(trainingSet[i],test,len(test)-1)
      distanceList.append((trainingSet[i],distance))
    distanceList.sort(key=operator.itemgetter(1))
    #print(distanceList)
    neighborList=[]
    for i in range(k):
        neighborList.append(distanceList[i][0])   #append the training set. 0th index indicates we are only appending the list and not the distance
    return neighborList

def getPredictions(neighbor):
    classification={}
    for i in range(len(neighbor)):
        result=neighbors[i][-1]   #class name
        if result in classification:
            classification[result]+=1
        else:
            classification[result]=1
    sortClassification=sorted(classification.items(),key=operator.itemgetter(1),reverse=True)  #get the most probable class.
    return sortClassification[0][0]   #return the most probable class.


def computeAccuracy(predictions,test):
    accuracy=0
    for i in range(0,len(test)):
        if(test[i][-1]==predictions[i]):
            accuracy+=1
    accuracy=float(accuracy)/len(test)*100
    return accuracy

if __name__ == '__main__':
    testSet, trainingSet=loadCsvFile()
    k=1
    predictedClass=[]
    for i in range(len(testSet)):
        neighbors=getSimilarity(trainingSet,testSet[i],k)
        result=getPredictions(neighbors)
        predictedClass.append(result)
    print(predictedClass)
    accuracyObtained=computeAccuracy(predictedClass,testSet)
    print("Accuracy of the kNN Classifier =" + str(accuracyObtained))