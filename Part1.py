import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ClassData5  = pd.read_table("Classification music(1)\Classification music\GenreClassData_5s.txt")
# ClassData10 = pd.read_table("Classification music(1)\Classification music\GenreClassData_10s.txt")
ClassData30 = pd.read_table("Classification music(1)\Classification music\GenreClassData_30s.txt")

# Divide data into testing sets and training sets

trainingSet30 = ClassData30.query("Type == 'Train'") # Training set
testingSet30 = ClassData30.query("Type == 'Test'")  # Testing set

# Task 1
    # (A) Design a k-NN classifier (with k=5) for all ten genres using only:
    # spectral_rolloff_mean, mfcc_1_mean, sepctral_centroid_mean and tempo
    # I.e find the 5 closest references and choose the class with the most 
    # references

k = 5
features = trainingSet30[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo', 'Genre']]
references = testingSet30[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo', 'Genre']]

