import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Part1 as p1

features = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
mapGenre = {"pop": 0, "disco": 1, "metal" : 2, "classical" : 3}


training = p1.trainingSet30[p1.trainingSet30.Genre.isin(mapGenre.keys())]
testing = p1.testingSet30[p1.testingSet30.Genre.isin(mapGenre.keys())]


errorRate, confusionMatrix = p1.NNclassifier(testing, training, mapGenre, features=features)
print("The error rate is:", errorRate)
print(confusionMatrix)