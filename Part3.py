import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Part1 import NNclassifier, trainingSet30, testingSet30, GENREMAP, ClassData30, ALLFEATURES
import time



FEATURES = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
# candidates = ['spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'spectral_rolloff_var']
featuresP3 = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'spectral_rolloff_var']
mapGenre = {"pop": 0, "disco": 1, "metal" : 2, "classical" : 3}


training = trainingSet30[trainingSet30.Genre.isin(GENREMAP.keys())]
testing = testingSet30[testingSet30.Genre.isin(GENREMAP.keys())]


def tryAll(features=ALLFEATURES, base=FEATURES, data=ClassData30, Gmap=GENREMAP):
    """tryAll tries the k-NN classifier for all possible combinations of features where
    the base list of features base replace one of its element with an element from the features list

    Parameters
    ----------
    features : list(str), optional
        A list of the features that is to replace one of the elements in base
    base : list(str), optional
        A list of the base features. 
    data : pd.Dataframe
        A pandas datafram that contains the relevant data
    Gmap : list(tuple(str, int)), optional
        List of the tuple (Genre, index). Genre is a string of a genre in data.
        index is an integer that corresponds to the row number of Genre in the 
        generated confusion matrix
    
    """
    start = time.time()
    error = {}
    for i in range(len(base)):
        for j in range(len(features)):
            f = base.copy()
            f[i] = features[j]
            errorRate, confusionMatrix = NNclassifier(data=data, features=f, mapIndex=Gmap)
            error[(base[i], features[j])] = (errorRate, confusionMatrix)
    end = time.time()
    print(end-start, 'Seconds')
    return error


error, confusionMatrix = NNclassifier(features=featuresP3)
print('Estimated error rate for Task 3:', error)
print('Confusion matrix for Task 3')
print(confusionMatrix)

# Output of tryAll for the four genres:
# Spectral_rolloff_var replacing tempo : 0.1392405063291139
# Matrix:
# [16 4 0 0]
# [3 14 2 1]
# [0 1 19 0]
# [0 0 0 19]

# Takes something like 90 min to run, so not recommended