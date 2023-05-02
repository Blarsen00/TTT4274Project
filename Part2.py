from Part1 import NNclassifier
from Part3 import FEATURES, mapGenre

feature1 = FEATURES.copy()
feature2 = FEATURES.copy()
feature3 = FEATURES.copy()
feature4 = FEATURES.copy()

feature1.remove('spectral_rolloff_mean')
feature2.remove('mfcc_1_mean')
feature3.remove('spectral_centroid_mean')
feature4.remove('tempo')

# print("Answer to Task 2a")
# error, CM = NNclassifier(features=feature1, mapIndex=mapGenre)
# print(error)
# print(CM)

# print()

# error, CM = NNclassifier(features=feature2, mapIndex=mapGenre)
# print(error)
# print(CM)

# print()

# error, CM = NNclassifier(features=feature3, mapIndex=mapGenre)
# print(error)
# print(CM)

# print()

# error, CM = NNclassifier(features=feature4, mapIndex=mapGenre)
# print(error)
# print(CM)