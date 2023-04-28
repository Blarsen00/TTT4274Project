import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum

# class Genre(Enum):
#     pop = 1


# ClassData5  = pd.read_table("Classification music(1)\Classification music\GenreClassData_5s.txt")
# ClassData10 = pd.read_table("Classification music(1)\Classification music\GenreClassData_10s.txt")
ClassData30 = pd.read_table("Classification music(1)\Classification music\GenreClassData_30s.txt")

# Divide data into testing sets and training sets
trainingSet30 = ClassData30.query("Type == 'Train'")    # Training set
testingSet30 = ClassData30.query("Type == 'Test'")      # Testing set

# Task 1
    # (A) Design a k-NN classifier (with k=5) for all ten genres using only:
    # spectral_rolloff_mean, mfcc_1_mean, sepctral_centroid_mean and tempo
    # I.e find the 5 closest references and choose the class with the most 
    # references

    # Training a NN classifier would be the same as choosing good references
    # The references in this case will be the entire training set. The distance
    # measure is the Euclidian dinstance

k = 5
# features = trainingSet30[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo', 'Genre']]
references = trainingSet30[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo', 'Genre']]

def euclidianDistance(x, u):
    return np.matmul(np.transpose(x-u), (x-u))


def NNclassifier(testSet, references, distance=euclidianDistance, k=5, 
    features=['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']):
    mapIndex = {"pop": 0, "disco": 1, "metal" : 2, "classical" : 3, "rock" : 4,
                "blues" : 5, "reggae" : 6, "hiphop" : 7, "country" : 8,
                "jazz" : 9}
    # closestPoints = [(np.inf, 'Genre')] * k
    # furthestPoint = np.inf

    # closestPoints = pd.DataFrame(
    #     {"Distance" : np.full((1,k), np.inf),
    #     "Genre"    : np.full((1,k), "Genre")})
    confusionMatrix = pd.DataFrame(0,index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        columns=["pop", "disco", "metal", "classical", "rock", "blues", "reggae", "hiphop", 
        "country", "jazz"])
    
    for i in range(len(testSet)):
        # closestPoints = pd.DataFrame(
        # {"Distance" : [np.inf] * k,
        # "Genre"    : ["Genre"] * k})
        # furthestPoint = closestPoints.max()
        # distances = []

        closestPoints = [(np.inf, 'Genre')] * k
        furthestPoint = (np.inf, 'Genre')
        
        x = np.array(testSet.iloc[i, [testSet.columns.get_loc(c) for c in features if c in testSet]])
        
        for j in range(len(references)):
            u = np.array(references.iloc[j, [references.columns.get_loc(c) for c in features if c in references]])
            d = distance(x, u)


            if d < furthestPoint[0]:
                closestPoints.remove(furthestPoint)
                closestPoints.append((d, references.iloc[j, references.columns.get_loc("Genre")]))
                # furthestPoint = (d, references.iloc[j, references.columns.get_loc("Genre")])
                furthestPoint = max(closestPoints, key = lambda t : t[0])

            # distances.append(d)
            # if d < furthestPoint["Distance"]:
            #     g = references.iat[i, references.columns.get_loc("Genre")]
            #     closestPoints.at[closestPoints.Distance == furthestPoint.Distance, 'Distance'] = d
            #     closestPoints.at[closestPoints.Distance == d, 'Genre'] = g
            #     # closestPoints.at[closestPoints["Distance"] == d, "Genre"] = g
            #     furthestPoint = closestPoints.max()

            # if d < furthestPoint:
            #     for k in range(len(closestPoints)):
            #         if closestPoints[k][0] == furthestPoint:
            #             closestPoints[k] = (d, testSet[j, ['Genre']])
            #             furthestPoint = np.max(closestPoints[0])
        
        # print(closestPoints)
        closestPoints.sort(key=lambda x : x[0])
        count = {}
        for z in range(len(closestPoints)):
            if closestPoints[z][1] in count.keys():
                count[closestPoints[z][1]] += 1
            else:
                count[closestPoints[z][1]] = 1

        classification = max(count, key=count.get)                  # Classified genre
        genre = testSet.iat[i, testSet.columns.get_loc("Genre")] # Actual genre

        confusionMatrix.at[mapIndex[genre], classification] += 1    # Update the confusion matrix
    
    correctClassifications = pd.Series(np.diag(confusionMatrix), index=[confusionMatrix.index, confusionMatrix.columns])
    num = correctClassifications.sum()
    errorRate = 1-num/confusionMatrix.sum().sum()


    return errorRate, confusionMatrix


    # return closestPoints


    
error, CM = NNclassifier(testSet=testingSet30, references=references)
print("The error rate is:", error)
print(CM)
    



 
